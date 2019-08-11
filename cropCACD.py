import openface
import os
import cv2
import sys  
import math
import PIL.Image as PILImage

dlib_path = "./shape_predictor_68_face_landmarks.dat"
pic_origin_path = "../DATA/CACD/CACD2000"
landmark_path = "../DATA/CACD/landmark"
pic_cropped_path = "../DATA/CACD/CACD_30_30_eyecroped"
# pic_cropped_path = "./after_crop"
#如果剪切目录不存在则创建
if not os.path.exists(pic_cropped_path):
    os.mkdir(pic_cropped_path) 

# 提取本文件所在目录
fileDir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.dirname(fileDir)) #为了引用不在同一个文件夹中的包

def load_landmark(landmark_path):
    try:
        file = open(landmark_path,'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    rows=len(content) #文件行数 16
    output = [] #初始化输出
    eye_location = []
    row_count=0
    for i in range(rows):
        list_temp = (content[i].strip().split('\t'))[0].split(" ")
        list_temp = tuple(map(float, list_temp))
        output.append(list_temp)
        if i == 1 or i == 0:
            eye_location.append(list_temp)
        row_count+=1
    file.close()
    return output, eye_location

def get_eye_location(img_dir,ldmark_dir,name):
    img_path = img_dir+"/"+name+".jpg"
    landmark_path = ldmark_dir+"/"+name+".landmark"
    after_read = cv2.imread(img_path)
    landmark_list,eye_location = load_landmark(landmark_path)
    return eye_location

def show_dlib_face(img_dir,ldmark_dir,name):
    img_path = img_dir+"/"+name+".jpg"
    landmark_path = ldmark_dir+"/"+name+".landmark"
    after_read = cv2.imread(img_path)
    landmark_list,eye_location = load_landmark(landmark_path)


    openface_alilgn = openface.AlignDlib(dlib_path) #实例化openface的AlignDlib类
    return_rect = openface_alilgn.getLargestFaceBoundingBox(after_read) #返回dlib.rectanngle
    print("Height: ",return_rect.height())
    print("Width: ",return_rect.width())
    # cropped = after_read[return_rect.left():return_rect.top(), return_rect.right():return_rect.bottom()]
    cv2.rectangle(after_read,(return_rect.left(),return_rect.top()),(return_rect.right(),return_rect.bottom()),(0,255,0),2)
    i = 1
    for point in landmark_list:
        point_int = (int(point[0]),int(point[1]))
        cv2.circle(after_read, point_int, 1, (0,0,255), 4)
        #cv2.putText(after_read, str(i), point_int, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,0,0), 1, 4)
        i+=1
    cv2.imshow("nice",after_read)
    cv2.waitKey(200000)

def Distance(p1,p2):  
    dx = p2[0]- p1[0]  
    dy = p2[1]- p1[1]  
    return math.sqrt(dx*dx+dy*dy)  

def ScaleRotateTranslate(image, angle, center =None, new_center =None, scale =None, resample= PILImage.BICUBIC):  
    """
    根据参数，求仿射变换矩阵和变换后的图像。
    
    resample:变换方式，采用双三次插值
    """
    if (scale is None )and (center is None ):  
        return image.rotate(angle=angle, resample=resample)  
    nx,ny = x,y = center  
    sx=sy=1.0  
    if new_center:  
        (nx,ny) = new_center  
    if scale:  
        (sx,sy) = (scale, scale)  
    cosine = math.cos(angle)  
    sine = math.sin(angle)  
    a = cosine/sx  
    b = sine/sx  
    c = x-nx*a-ny*b  
    d =-sine/sy  
    e = cosine/sy  
    f = y-nx*d-ny*e  
    return image.transform(image.size, PILImage.AFFINE, (a,b,c,d,e,f), resample=resample)

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):  
    # 根据所给的人脸图像，眼睛坐标位置，偏移比例，输出的大小，来进行裁剪。
    # calculate offsets in original image 计算在原始图像上的偏移。  
    offset_h = math.floor(float(offset_pct[0])*dest_sz[0])  
    offset_v = math.floor(float(offset_pct[1])*dest_sz[1])  
    # get the direction  计算眼睛的方向。  
    eye_direction = (eye_right[0]- eye_left[0], eye_right[1]- eye_left[1])  
    # calc rotation angle in radians  计算旋转的方向弧度。  
    rotation =-math.atan2(float(eye_direction[1]),float(eye_direction[0]))  
    # distance between them  # 计算两眼之间的距离。  
    dist = Distance(eye_left, eye_right)  
    # calculate the reference eye-width    计算最后输出的图像两只眼睛之间的距离。  
    reference = dest_sz[0]-2.0*offset_h  
    # scale factor   # 计算尺度因子。  
    scale =float(dist)/float(reference)  
    # rotate original around the left eye  # 原图像绕着左眼的坐标旋转。  
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)  
    # crop the rotated image  # 剪切  
    crop_xy = (eye_left[0]- scale*offset_h, eye_left[1]- scale*offset_v)  # 起点  
    crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)   # 大小  
    image = image.crop((int(crop_xy[0]),int(crop_xy[1]),int(crop_xy[0]+crop_size[0]),int(crop_xy[1]+crop_size[1])))  
    # resize it 重置大小  
    image = image.resize(dest_sz, PILImage.ANTIALIAS)  
    return image  

if __name__ == "__main__":
    i = 0
    for filename in os.listdir(pic_origin_path):
        print(i,"  file name without .jpg:",filename[:-4])
        image = PILImage.open(pic_origin_path+'/'+filename)
        eye_location = get_eye_location(pic_origin_path,landmark_path,filename[:-4])
        leftx = int(eye_location[0][0])
        lefty = int(eye_location[0][1])
        rightx = int(eye_location[1][0])
        righty = int(eye_location[1][1])
        # show_dlib_face(pic_origin_path,landmark_path,filename[:-4])
        CropFace(image, eye_left=(leftx,lefty), eye_right=(rightx,righty), offset_pct=(0.3,0.3), dest_sz=(200,200)).save(pic_cropped_path+'/'+filename) 
        i = i+1