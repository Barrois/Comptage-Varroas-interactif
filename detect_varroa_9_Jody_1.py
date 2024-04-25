#!/usr/bin/env python
import cv2
import os
import sys, getopt
import numpy as np
from edge_impulse_linux.image import ImageImpulseRunner

runner = ImageImpulseRunner("./comptage-de-varroas-linux-x86_64-v18.eim")

"""
attention il faut rendre le fichier "eim" executable
Use the ls -l ./comptage-de-varroas-linux-x86_64-v15.eim command to view the file permissions.
If the file doesn't have execute permissions for the owner (indicated by -rw-r--r--),
add execute ./comptage-de-varroas-linux-x86_64-v19.eim.
"""

model_info = runner.init()
print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
labels = model_info['model_parameters']['labels']

folder_filtre_a_crop= "./filtre_a_crop/"
# folder_filtre_a_crop= "/home/ubuntu/websites/VarroaCounter/public/filtre_a_crop/"
folder_traite = "./traite/"
folder_sauve = "./sauve/"
folder_sauve_crop_Eloquent = "/home/ubuntu/websites/VarroaCounter/public/sauve_crop_Eloquent/"
folder_sauve_pas_varroas = "/home/ubuntu/websites/VarroaCounter/public/sauve_pas_varroas/"
folder_contours = "./contours/"

for filename in os.listdir(folder_contours):
    os.remove(os.path.join(folder_contours, filename))  # supprime tous les fichiers du dossier folder_contours /contours/


morph_radius_min = 4      # morph_radius = 7
morph_radius_max = 8      # 7

image_hue_min = 0
image_hue_max = 80   # 50
image_sat_min = 0    # 25
image_sat_max = 255
image_val_min = 0
image_val_max = 190 # 180

vorroa_area_min = 100  # 150
vorroa_area_max = 700  # 250
vorroa_width_min = 10   # 10
vorroa_width_max = 40  # 20
vorroa_height_min = 10  # 10
vorroa_height_max = 40 # 20
vorroa_ratio_min =  0.2  # 0.6
vorroa_ratio_max =  0.95  # 0.8

h1 = 32  # taille du crop  de 32 x 32
lim_vrai = 0.5 # limite de vraissemblance  0.90


for filename in os.listdir(folder_sauve_crop_Eloquent) :
    if filename != "index.php":
        os.remove(os.path.join(folder_sauve_crop_Eloquent, filename))  # supprime les fichiers de /sauve_varroas/
for filename in os.listdir(folder_sauve_pas_varroas) :
    if filename != "index.php":
        os.remove(os.path.join(folder_sauve_pas_varroas, filename))  # supprime les fichiers de /sauve_crop/


filename1 = "N_20240219_24.jpg"
folder0 = "./neo_scan_Michel/"

# nom court = nom de l'image sans jpg
c = filename1
cc= len(c)
nom_court = c[0:cc-4]
for filename in os.listdir(folder_filtre_a_crop) :
    os.remove(os.path.join(folder_filtre_a_crop, filename))  # supprime les fichiers de /filtre_a_crop/
fname = folder0+filename1
print (fname)
compteur_varroas = 0
im = cv2.imread(fname)           # image d'origine
w_image = im  # sauvegarde de l'image
c_image = im
cpt_varroa = int(0)
CX=[]
CY=[]
# masque HSV Eloquent
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
hsv_orig = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
hue = hsv[:, :, 0]
sat = hsv[:, :, 1]
val = hsv[:, :, 2]
logs = []
# mask non-interesting parts of image
mask1 = (hue <= image_hue_max) & (hue >= image_hue_min)
hue[mask1] = 255
hue[~mask1] = 0
mask2 = (sat <= image_sat_max) & (sat >= image_sat_min)
sat[mask2] = 255
sat[~mask2] = 0
mask3 = (val <= image_val_max) & (val >= image_val_min)
val[mask3] = 255
val[~mask3] = 0
mask = mask1 & mask2 & mask3
mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((morph_radius_min, morph_radius_max)))
res = im.copy()
res[mask < 1] = 0
# find contours
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 1, 255, 0)
contours, _ = cv2.findContours(thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
# image avec les contours
contoured_image = cv2.drawContours(c_image.copy(), contours, -1, (0, 255, 0), 2)
contoured_image_name = folder_contours +str(len(contours))+"_"+ filename1
cv2.imwrite(contoured_image_name,contoured_image) # ecrit l'image_final sur disque dans le folder_traite /traite/
print(" nb de contours dans le masque ",len(contours),contoured_image_name,len(contoured_image) )
# on part de la liste des contours
for cnt in contours:
    if cnt.shape[0] < 200 :  # on limite les shape entre 300 et 50
        if cnt.shape[0] < 5:# 5  minimum
            continue
        ((cx, cy), (width, height), angle) = cv2.fitEllipse(cnt)
        try:
            cx, cy = int(cx), int(cy)
            width, height = int(width), int(height)
        except ValueError:
            continue
        area = cv2.contourArea(cnt)
    # logs.append(f'CANDIDATe: cx={cx}, cy={cy}, width={width}, height={height}, area={area}')
    # print(f'CANDIDATe: cx={cx}, cy={cy}, width={width}, height={height}, area={area}')
    # rajout
    if (vorroa_width_min <= width <= vorroa_width_max):
        if (vorroa_height_min <= height <= vorroa_height_max):
            if (vorroa_area_min <= area <= vorroa_area_max):
                ratio = min(width, height) / max(width, height)
                if (vorroa_ratio_min< ratio <vorroa_ratio_max):
                    # print('...ok because of Area, Width, Height and Ratio', compteur_varroas)
                    # res2 = cv2.ellipse(im2, (cx, cy), (width // 2, height // 2), angle, 0, 360, (0, 0, 255), 2)
                    # name_res2 = folder_sauve+'res2.jpg'
                    # cv2.imwrite(name_res2,res2)
                    # print(" compteur_varroas ",compteur_varroas)
                    str_cx = "{:05d}".format(cx) # sur 5 caractères
                    str_cy = "{:05d}".format(cy) # sur 5 caractères
                    # name_crop = folder_filtre_a_crop +'crop_'+str_cx+"_"+str_cy+"_"+str(compteur_varroas)+'_'+nom_court+'.jpg'
                    h = int( h1/2 )
                    crop_img = w_image[cy-h:cy+h,cx-h:cx+h]
                    # im2 = cv2.rectangle(im, (cx-h,cy-h), (cx+h,cy+h), (0,0,255), 3) # le rectangle est en coin haut coin bas

                    if crop_img.size != 0:   # Do something if crop_img has a value
                        # print(" name_crop ", name_crop, cx-h,cx+h,cy-h,cy+h, crop_img.size, cx, cy, width ,height  )
                        # cv2.imwrite(name_crop,crop_img) # ecrit le crop sur disque dans le folder_filtre_a_crop  : filtre_a_crop
                        compteur_varroas = compteur_varroas + 1
                        # là commence l'IA
                        features, cropped = runner.get_features_from_image(crop_img)
                        res = runner.classify(features)
                        if "classification" in res["result"].keys():
                                # print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                                for label in labels:
                                    score = res['result']['classification'][label]
                                    # print('%s: %.2f\t' % (label, score), end='')

                                    if label == "varroa": # Varroa
                                        if score > lim_vrai :  # limite de vraissemblance
                                            str_score = "{:02d}".format(int(score*100))
                                            print ("varroa discrimine score 0.",str_score,"  ", filename1," cx ",cx," cy ",cy)
                                            CX.append(cx)
                                            CY.append(cy)
                                            # nom_du_crop = folder_sauve_crop_Eloquent + str_score + "_"+ nom_court +"_"+filename
                                            nom_du_crop = folder_sauve_crop_Eloquent + str_score +"_"+filename1
                                            # print(nom_du_crop)
                                            cv2.imwrite(nom_du_crop, crop_img) # dans sauve_crop pour usage ultérieur : tri des varroas etc...
                                            cpt_varroa = cpt_varroa + 1

                                continue
im1 = cv2.imread(fname)
h = int( h1/2 )
print(' nb total de varroas discrimines ',cpt_varroa)
if cpt_varroa !=0:
    for i in range(cpt_varroa) :
        im1 = cv2.rectangle(im1, (CX[i]-h,CY[i]-h), (CX[i]+h,CY[i]+h), (0,0,255), 3)
font = cv2.FONT_HERSHEY_SIMPLEX
org = (150, 100)
fontScale = 3
color = (255, 0, 0)
thickness = 5
text = "Nombre de varroas detectes : "+str(cpt_varroa)
img2 = cv2.putText(im1, text, org, font, fontScale, color, thickness)
image_final = folder_traite + filename1
cv2.imwrite(image_final,img2) # ecrit l'image_final sur disque dans le folder_traite /traite/
print(" nom de l image", image_final)
