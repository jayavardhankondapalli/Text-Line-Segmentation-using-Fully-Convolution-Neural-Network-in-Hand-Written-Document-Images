import tensorflow as tf
import os
import cv2
import queue
import numpy as np
from numpy import array
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
patches_cc = 0
img_no = "018"
def partitions():
    img_name = img_no+'.tif'
    l1 = [img_name]
    for imgname in l1:
        x = 0
        y = 0
        id1 = 0
        height = 168
        width = 328
        img = cv2.imread(imgname, 0)
        shape = img.shape
        imgname = imgname.strip('.tif')
        patch = []
        l = -1
        for i in range(0, height):
            patch.append([])
            l += 1
            for j in range(0, width):
                patch[i].append(255)
        patch = np.array(patch)

        tx = 0
        ty = 0
        hor_patch = int(shape[1] / width)
        ver_patch = int(shape[0] / height)

        for i in range(0, ver_patch):
            tx = 0
            for j in range(0, hor_patch):
                ti = 0
                for k in range(ty, ty + height):
                    tj = 0
                    for l in range(tx, tx + width):
                        patch[ti][tj] = img[k][l]
                        tj += 1
                    ti += 1
                tx = tx + width
                name = str(imgname) + '_' + str(id1) + '.tif'
                cv2.imwrite(name, patch)
                id1 += 1
            for i in range(0, height):
                for j in range(0, width):
                    if 'z' in imgname:
                        patch[i][j] = 0
                    else:
                        patch[i][j] = 255
            ti = 0
            for k in range(ty, ty + height):
                tj = 0
                for l in range(tx, shape[1]):
                    patch[ti][tj] = img[k][l]
                    tj += 1
                ti += 1

            name = str(imgname) + '_' + str(id1) + '.tif'
            cv2.imwrite(name, patch)
            id1 += 1
            ty += height
        # wewe

        tx = 0
        for j in range(0, hor_patch):
            for i in range(0, height):
                for j in range(0, width):
                    patch[i][j] = 255
            ti = 0
            for k in range(ty, shape[0]):
                tj = 0
                for l in range(tx, tx + width):
                    patch[ti][tj] = img[k][l]
                    tj += 1
                ti += 1
            tx = tx + width
            name = str(imgname) + '_' + str(id1) + '.tif'
            cv2.imwrite(name, patch)
            id1 += 1
        for i in range(0, height):
            for j in range(0, width):
                patch[i][j] = 255
        ti = 0
        for k in range(ty, shape[0]):
            tj = 0
            for l in range(tx, shape[1]):
                patch[i][tj] = img[k][l]
                tj += 1
            ti += 1
        name = str(imgname) + '_' + str(id1) + '.tif'
        cv2.imwrite(name, patch)
    print("Partitions completed")
    print("Number of patches are ",id1)
    return id1


def convol():
    class convolution():
        def __init__(self,kernal_size,input_filters,output_filters):
            self.W = tf.Variable(tf.random_normal([kernal_size,kernal_size,input_filters,output_filters],stddev=0.05))
        def feedforward(self,input1,stride=1):
            self.input = input1
            self.layer = tf.nn.conv2d(input1,self.W,strides=[1,stride,stride,1],padding='SAME')
            self.layerA = tf.nn.relu(self.layer)
            return self.layerA
    class deconvolution():
        def __init__(self,kernal_size,input_filters,output_filters):
            self.W = tf.Variable(tf.random_normal([kernal_size,kernal_size,input_filters,output_filters],stddev=0.05))
            self.input_filters = input_filters
        def feedforward(self,input1,stride=1,output_width=1,output_height=1):
            self.input = input1
            current_shape_size = input1.shape
            self.layer = tf.nn.conv2d_transpose(input1, self.W,
                                                    output_shape=[1] + [int(current_shape_size[1].value * 2),
                                                                       int(current_shape_size[2].value * 2),
                                                                       int(self.input_filters)], strides=[1, 2, 2, 1],
                                                    padding='SAME')

            self.layerA = tf.nn.relu(self.layer)
            return self.layerA

    X = tf.placeholder(shape=[None,168,328,1],dtype=tf.float32)
    Y = tf.placeholder(shape=[None,168,328,2],dtype=tf.float32)

    l1 = convolution(3,1,64)
    l2 = convolution(3,64,64)

    l3 = convolution(3,64,128)
    l4 = convolution(3,128,128)

    l5 = convolution(3,128,256)
    l6 = convolution(3,256,256)
    l7 = convolution(3,256,256)

    l8 = deconvolution(7,2048,256)

    l9 = deconvolution(1,2048,2048)
    l10 = deconvolution(1,2,2048)



    layer1 = l1.feedforward(X)
    layer2 = l2.feedforward(layer1)
    layer3_input = tf.nn.max_pool(layer2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

    layer3 = l3.feedforward(layer3_input)
    layer4 = l4.feedforward(layer3)
    layer5_input = tf.nn.max_pool(layer4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

    layer5 = l5.feedforward(layer5_input)
    layer6 = l6.feedforward(layer5)
    layer7 = l7.feedforward(layer6)
    layer8_input = tf.nn.max_pool(layer7,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

    layer8 = l8.feedforward(layer8_input)
    layer9 = l9.feedforward(layer8)
    layer10 = l10.feedforward(layer9)

    cost = tf.reduce_mean(tf.square(layer10-Y))
    auto_train = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver()

    saver.restore(sess,"save100.ckpt")

    """
    # 328 * 168
    train_images_names=['001.tif', '002.tif', '004.tif', '005.tif', '011.tif', '012.tif', '013.tif', '014.tif', '015.tif', '016.tif', '018.tif', '019.tif', '020.tif', '021.tif', '022.tif', '023.tif', '024.tif', '026.tif', '031.tif', '032.tif', '033.tif', '034.tif', '035.tif', '036.tif', '037.tif', '038.tif', '039.tif', '040.tif', '041.tif', '042.tif', '043.tif', '044.tif', '045.tif', '046.tif', '047.tif', '049.tif', '050.tif', '048.tif', '051.tif', '052.tif', '053.tif', '054.tif', '061.tif', '062.tif', '063.tif', '064.tif', '065.tif', '071.tif', '072.tif', '073.tif', '074.tif', '075.tif', '076.tif', '077.tif', '078.tif', '079.tif', '080.tif', '081.tif', '082.tif', '083.tif', '084.tif', '085.tif', '086.tif', '087.tif', '088.tif', '089.tif', '091.tif', '092.tif', '093.tif', '094.tif', '095.tif', '096.tif', '097.tif', '098.tif', '099.tif', '100.tif', '111.tif', '113.tif', '114.tif', '115.tif', '116.tif', '117.tif', '118.tif', '119.tif', '120.tif','141.tif', '142.tif', '143.tif', '144.tif', '145.tif', '151.tif', '152.tif', '153.tif', '154.tif', '157.tif', '158.tif', '159.tif', '160.tif', '162.tif', '163.tif', '165.tif', '167.tif', '169.tif', '170.tif', '171.tif', '173.tif', '174.tif', '177.tif', '179.tif', '180.tif', '192.tif', '193.tif']
    train_images=[]
    train_labels=[]
    patches_count = []
    with open('patches_count.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row!=[]:
                patches_count.append(row)
    
    for imgname in train_images_names:
        for_save = imgname
        patch_index = 0
        patch_count=0
        for k in range(0, len(patches_count)):
            if patches_count[k][0] == imgname:
                patch_index = k
                break
        patch_count = int(patches_count[patch_index][1])
        print(patch_count)
        for i in range(patch_count+1):
            patch_name = imgname.strip('.tif')+'_'+str(i)+'.tif'
            print(patch_name)
            patch_img = cv2.imread(patch_name,0)
            patch_img = np.expand_dims(patch_img, axis=2)
            train_images.append(patch_img)
    
            lpatch_name = 'z'+imgname.strip('.tif')+'_l_'+str(i)+'.tif'
            bpatch_name = 'z'+imgname.strip('.tif')+'_b_'+str(i)+'.tif'
            print(lpatch_name,bpatch_name)
            lpatch_img = cv2.imread(lpatch_name, 0)
            bpatch_img = cv2.imread(bpatch_name, 0)
            label=np.ones((168,328,2))
            for ii in range(168):
                for jj in range(328):
                    t = [lpatch_img[ii][jj],bpatch_img[ii][jj]]
                    label[ii][jj] = t
    
            input_data={X:[patch_img],Y:[label]}
            aa = sess.run([cost,auto_train,l1.W,l4.W,l10.W], feed_dict=input_data)
            print("cost ",aa[0])
            print("l1 ",aa[2])
            print("l4 ",aa[3])
            print("l10",aa[4])
            train_labels.append(label)
        if for_save=="001.tif":
            save_path = saver.save(sess, "asave/save1.ckpt")
            print("Save to path: ", save_path)
        elif for_save=="004.tif":
            save_path = saver.save(sess, "asave/save4.ckpt")
            print("Save to path: ", save_path) 
        elif for_save=="016.tif":
            save_path = saver.save(sess, "asave/save16.ckpt")
            print("Save to path: ", save_path)
        elif for_save=="042.tif":
            save_path = saver.save(sess, "asave/save42.ckpt")
            print("Save to path: ", save_path)
        elif for_save=="073.tif":
            save_path = saver.save(sess, "asave/save73.ckpt")
            print("Save to path: ", save_path)
        elif for_save=="100.tif":
            save_path = saver.save(sess, "asave/save100.ckpt")
            print("Save to path: ", save_path)
        elif for_save=="169.tif":
            save_path = saver.save(sess, "asave/save169.ckpt")
            print("Save to path: ", save_path)
        elif for_save=="193.tif":
            save_path = saver.save(sess, "asave/save193.ckpt")
            print("Save to path: ", save_path)
    """
    print("Convolution neural network started")
    temp = patches_cc + 1
    for pt in range(temp):
        imgname = img_no+'_'+str(pt)+'.tif'
        img = cv2.imread(imgname,0)
        print(pt)
        img = np.expand_dims(img, axis=2)
        img=[img]
        input_data = {X:img}
        bb = sess.run(layer10,feed_dict=input_data)
        limg = np.ones((168,328))
        bimg = np.ones((168,328))
        for k in range(0,len(bb)):
            for i in range(168):
                for j in range(328):
                    limg[i][j] = bb[k][i][j][0]
                    bimg[i][j] = bb[k][i][j][1]
            cv2.imwrite("aline"+str(pt)+".tif",limg)
            cv2.imwrite("aback"+str(pt)+".tif",bimg)
    print("Convolution neural network completed")



def mix():
    print("Mixing started")
    img_name = img_no+".tif"
    img = cv2.imread(img_name,0)
    shape = img.shape
    wid = shape[1]
    hei = shape[0]
    pwid = 328
    phei = 168
    ans = []
    patch_count = patches_cc
    num_wid = wid/pwid
    if wid % pwid !=0:
        num_wid=int(num_wid)
        num_wid+=1

    nrows = int((patch_count+1)/num_wid)

    k=-1
    w = num_wid * pwid
    h = nrows * phei
    print(w,h)
    for i in range(h):
        ans.append([])
        k+=1
        for j in range(w):
            ans[k].append(0)
    ans = np.array(ans)

    cc=0

    for i in range(nrows):
        for j in range(num_wid):
            imgname = "aline"+str(cc)+".tif"
            cc+=1
            patch = cv2.imread(imgname,0)
            pi=0
            pj=0
            for m in range(i*phei,(i*phei)+phei):
                for n in range(j*pwid,(j*pwid)+pwid):
                    ans[m][n] = patch[pi][pj]
                    pj+=1
                pi+=1
                pj=0
    cv2.imwrite("real_line.tif",ans)
    print("Mixing completed")

def samp():
    img = cv2.imread('real_line.tif',0)
    cv2.imwrite("a.tif",img)

    shape = img.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img[i][j]>100:
                img[i][j] = 255
            else:
                img[i][j] = 0
    kernal = np.ones((5,5),np.uint8)
    img = cv2.dilate(img,kernal,iterations=2)
    cv2.imwrite("zlinedil2.tif",img)
    print("image enhancing completed")

def getstart(start,point):
    if start[1] > point[1]:
        start = point
    elif start[1] == point[1] and start[0] > point[0]:
        start = point
    return start

def getend(end,point):
    if point[1] > end[1]:
        end = point
    elif point[1] == end[1] and point[0] > end[0]:
        end = point
    return end



def components_and_lines():
    print("Extracting components")
    visited = []
    components = []
    ind = -1

    img = cv2.imread('zlinedil2.tif', 0)

    shape = img.shape

    reimg = []
    ree = []
    ind = -1
    for i in range(shape[0]):
        reimg.append([])
        ree.append([])
        ind += 1
        for j in range(shape[1]):
            reimg[ind].append([0, 0, 0])
            ree[ind].append(0)

    ind = -1

    for i in range(shape[0]):
        for j in range(shape[1]):
            if img[i][j] > 200:
                img[i][j] = 255
            else:
                img[i][j] = 0

    for i in range(shape[0]):
        visited.append([])
        ind += 1
        for j in range(shape[1]):
            visited[ind].append(0)

    lines = []
    start_comp = int(shape[1] / 3)
    for i in range(shape[1]):
        for j in range(shape[0]):
            if visited[j][i] == 0 and img.item(j, i) == 255:
                component = []
                que = queue.Queue(0)
                start = [shape[0], shape[1]]
                end = [0, 0]
                area = 0
                que.put([j, i])
                visited[j][i] = 1
                while que.empty() == False:
                    point = que.get()
                    component.append(point)
                    area += 1
                    start = getstart(start, point)
                    end = getend(end, point)

                    if point[0] + 1 < shape[0] and 255 == img.item(point[0] + 1, point[1]) and visited[point[0] + 1][
                        point[1]] == 0:  # +1 0
                        p = [point[0] + 1, point[1]]
                        visited[p[0]][p[1]] = 1
                        que.put(p)

                    if point[1] + 1 < shape[1] and 255 == img.item(point[0], point[1] + 1) and visited[point[0]][
                                point[1] + 1] == 0:  # 0 +1
                        a = [point[0], point[1] + 1]
                        visited[a[0]][a[1]] = 1
                        que.put(a)

                    if point[0] - 1 >= 0 and 255 == img.item(point[0] - 1, point[1]) and visited[point[0] - 1][
                        point[1]] == 0:  # -1 0
                        p = [point[0] - 1, point[1]]
                        visited[p[0]][p[1]] = 1
                        que.put(p)

                    if point[1] - 1 >= 0 and 255 == img.item(point[0], point[1] - 1) and visited[point[0]][
                                point[1] - 1] == 0:  # 0 -1
                        a = [point[0], point[1] - 1]
                        visited[a[0]][a[1]] = 1
                        que.put(a)

                if area < 300:
                    continue

                comp = {'start': start, 'end': end, 'area': area, 'points': component}

                components.append(comp)
                completed = 0
                if len(lines) == 0:
                    lines.append([comp])
                    completed = 1
                elif completed == 0:
                    if lines[0][len(lines[0]) - 1]['end'][0] >= comp['start'][0]:
                        if lines[0][len(lines[0]) - 1]['end'][0] - comp['end'][0] >= 50 and comp['start'][1] < start_comp:
                            lines.insert(0, [comp])
                            completed = 1
                        else:
                            if lines[0][len(lines[0]) - 1]['end'][1] < comp['start'][1]:
                                lines[0].append(comp)
                                completed = 1
                    else:
                        for i in range(0, len(lines) - 1):
                            lasti = lines[i][len(lines[i]) - 1]
                            lasti1 = lines[i + 1][len(lines[i + 1]) - 1]

                            if lasti['end'][0] <= comp['start'][0] and comp['start'][0] <= lasti1['end'][0]:
                                if comp['start'][0] - lasti['end'][0] >= 50 and lasti1['end'][0] - comp['start'][
                                    0] >= 50 and comp['start'][1] < start_comp:
                                    lines.insert(i + 1, [comp])
                                    completed = 1
                                elif comp['start'][0] - lasti['end'][0] < lasti1['end'][0] - comp['start'][0]:
                                    if lasti['end'][1] < comp['start'][1]:
                                        lines[i].append(comp)
                                        completed = 1
                                else:
                                    if lasti1['end'][1] < comp['start'][1]:
                                        lines[i + 1].append(comp)
                                        completed = 1
                                break

                    la = lines[len(lines) - 1][len(lines[len(lines) - 1]) - 1]['end'][0]
                    if la < comp['start'][0] and completed == 0 and comp['start'][1] < start_comp:
                        if comp['start'][0] - la >= 50:
                            lines.append([comp])
                        else:
                            lines[len(lines) - 1].append(comp)

    remove_list = []
    for i in range(len(lines)):
        if len(lines[i]) < 2:
            remove_list.append(i)
    for i in remove_list:
        lines.remove(lines[i])

    print("Number of components are ",len(components))
    print("Number of lines are ",len(lines))

    """
    for i in lines:
        print("new line")
        for j in i:
            print(j['start'])
    """

    reimg = np.array(reimg)
    sa = 0
    for line in lines:
        if sa == 0:
            b, g, r = 255, 0, 0
            sa += 1
        elif sa == 1:
            b, g, r = 0, 255, 0
            sa += 1
        elif sa == 2:
            b, g, r = 0, 0, 255
            sa = 0

        for components in line:
            for point in components['points']:
                reimg[point[0]][point[1]][0] = b  # b
                reimg[point[0]][point[1]][1] = g  # g
                reimg[point[0]][point[1]][2] = r  # r

    cv2.imwrite("zzz.tif", reimg)

    line_points = []
    mid_points = []
    ind = -1
    for line in lines:
        line_points.append([])
        ind += 1
        line.insert(0, {'start': [line[0]['start'][0], 0], 'end': [line[0]['start'][0], 0]})
        line.append({'start': [line[len(line) - 1]['end'][0], shape[1]], 'end': [line[len(line) - 1]['end'][0], shape[1]]})
        for i in range(1, len(line)):
            for ik in range(2):
                if ik == 0:
                    p1 = line[i - 1]['start']
                    p2 = line[i - 1]['end']
                else:
                    p1 = line[i - 1]['end']
                    p2 = line[i]['start']
                for y in range(p1[1], p2[1]):
                    x = ((((y - p1[1]) / (p2[1] - p1[1])) * (p2[0] - p1[0])) + p1[0])
                    x = int(x)
                    line_points[ind].append(x)
    ind = -1
    for i in range(1, len(line_points)):
        line1 = line_points[i - 1]
        line2 = line_points[i]
        mid_points.append([])
        ind += 1
        for j in range(0, shape[1]):
            mid_points[ind].append(int((line2[j] + line1[j]) / 2))

    print("Line segmentation Started")
    ori = cv2.imread(img_no+".tif", 0)
    final = cv2.imread(img_no+".tif", 1)
    shape = ori.shape

    lines_patches = []
    for i in range(len(lines)):
        lines_patches.append([])
        for j in range(300):
            lines_patches[i].append([])
            for k in range(shape[1]):
                lines_patches[i][j].append(255)


    for j in range(shape[1]):
        patch_line_ind = 0
        mid_ind = 0
        sa = 0
        b, g, r = 0, 0, 255
        for i in range(shape[0]):
            if mid_ind < len(lines) - 1 and i > mid_points[mid_ind][j]:
                patch_line_ind = 0
                if sa == 0:
                    b, g, r = 255, 0, 0
                    sa += 1
                elif sa == 1:
                    b, g, r = 0, 255, 0
                    sa += 1
                elif sa == 2:
                    b, g, r = 0, 0, 255
                    sa = 0
                mid_ind += 1
            if ori[i][j] == 0:
                final[i][j][0] = b
                final[i][j][1] = g
                final[i][j][2] = r
               # lines_patches[mid_ind][patch_line_ind][j] = 0
                #print(mid_ind,patch_line_ind,j,lines_patches[mid_ind][patch_line_ind][j])
                patch_line_ind+=1
    cv2.imwrite("final.tif", final)

    z_l =[]
    l_l = []
    num = shape[0]
    for i in range(shape[1]):
        z_l.append(0)
        l_l.append(num)
    mid_points.insert(0,z_l)
    mid_points.append(l_l)

    ra = len(mid_points)-1
    for i in range(ra):
        for j in range(shape[1]):
            for k in range(mid_points[i][j],mid_points[i+1][j]):
                if ori[k][j] == 0:
                    lines_patches[i][k-mid_points[i][j]][j] = 0



    for i in range(len(lines)):
        line_img =  np.array(lines_patches[i])
        na = "line"+str(i+1)+".tif"
        cv2.imwrite(na,line_img)
        #print(na,line_img)
    """
    for line in lines:
        line.insert(0,{'end':[line[0]['start'][0],0]})
        line.append({'start':[line[len(line)-1]['end'][0],shape[1]]})
        for i in range(1,len(line)):
            p1 = line[i-1]['end']
            p2 = line[i]['start']
            for y in range(p1[1],p2[1]):
                x = ((( (y-p1[1])  / (p2[1]-p1[1]) )*(p2[0]-p1[0]))+p1[0])
                x = int(x)
                ree[x][y] = 255
    
    
    ree = np.array(ree)
    cv2.imwrite("ree.tif",ree)
    """
    print("Line segmentation Completed")

patches_cc = partitions()
convol()
mix()
samp()
components_and_lines()
