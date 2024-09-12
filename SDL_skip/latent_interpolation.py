# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:54:51 2024

@author: Michael
"""

from PIL import Image#
import os
import glob

inter = False
#folder = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Output\Interpolation"
folder = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Output\SDL_skip\asym\flatten\170"


def get_concat_h(im1, im2):

    
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):

    
    dst = Image.new('RGB', (im1.width, im1.height+ im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def concat(inter, folder):
    Name=[]
    os.chdir(folder)
    for file in glob.glob("*.png"):
        Name.append(file.partition(".")[0])

    Path=[folder +"\\"+name+".png" for name in Name]
    
    if inter:
        i=0
        for path in Path:
            img = Image.open(path)
            left = 300
            top = 200
            right = 800
            bottom = 800
            img = img.crop((left, top, right, bottom))
            
            if i == 0:
                img_prev = img
            else:
                img_prev = get_concat_h(img_prev, img)
            i+=1
        name = "\\interpolation.jpg"
    else:
        i=0
        for path in Path:
            img = Image.open(path)
            left = 530
            top = 180
            right = 1070
            bottom = 820
            img = img.crop((left, top, right, bottom))
            
            if i == 0:
                img_prev = img
                
            elif (i+1)%10==0:
                if i == 9:
                    img_prev_v = img_prev
                img_prev_v = get_concat_v(img_prev_v, img_prev)
                img_prev = img
                
            else:
                img_prev = get_concat_h(img_prev, img) 
    
            i+=1
        name = "\\sample.jpg"
        img_prev = img_prev_v 
        
    img_prev.show()
    img_prev.save(folder + name)
    
concat(inter, folder)