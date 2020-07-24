from distortion import Distortion
from distortion import add_imperfections
import cv2
import constants as c
from utility import urand 
import random
import numpy as np
from copy import copy

class DistortionWheel:
    def __init__(self,xdim,ydim,use_color=True, debug=False):
        """remember xdim and ydim are flipped >_<"""
        self.use_color = use_color
        self.debug = debug
        self.warp_distort = Distortion(xdim,ydim, disable=False) # problems
        self.wave_distort = Distortion(xdim,ydim, disable=False) # problems
        self.scale_distort = Distortion(xdim,ydim, disable=False) # problems
        self.rot_distort = Distortion(xdim,ydim, disable=False) # problems
        self.displace = Distortion(xdim,ydim, disable=False)# problems
        self.make_priorities()
        self.initialize_distortions()
        
    def set_warp(self):
        if self.debug:
            print('Setting warp')
        
        self.warp_distort.create_sinusoidal_warp(
            urand(c.warp_ax),
            urand(c.warp_ay),
            urand(c.warp_perx),
            urand(c.warp_pery),
            urand(c.warp_phax),
            urand(c.warp_phay)
        )
        
    def set_wave(self):
        if self.debug:
            print('Setting wave')
            
        self.wave_distort.create_sinusoidal_wave(
            urand(c.wave_ax),
            urand(c.wave_ay),
            urand(c.wave_perx),
            urand(c.wave_pery),
            urand(c.wave_phax),
            urand(c.wave_phay)
        )
        
    def set_tint(self):
        if self.debug:
            print('Setting tint')
            
        if self.use_color:
            self.tint = [urand(c.rgb_shift) for _ in range(3)]
        else:
            self.tint = [0,0,0]
        
    def set_scale(self):
        if self.debug:
            print('Setting scale')
            
        self.scx = urand(c.scale_x)
        self.scy = urand(c.scale_y)
        self.scale_distort.calculate_scale(
            (self.scx,self.scy),
            offset=(
                urand(c.scale_x_offset),
                urand(c.scale_y_offset)
                )
            )

    def set_rotation(self):
        if self.debug:
            print('Setting rotation')
            
        self.rot_distort.calculate_rotation(
            urand(c.rot_theta),
            offset = (
                urand(c.rot_offset_x),
                urand(c.rot_offset_y)
                )
            )

    def set_offset(self):
        if self.debug:
            print('Setting offset')
            
        self.displace.create_affine(
            1.,
            1.,
            0,
            0,
            urand(c.x_offset),
            urand(c.y_offset)
        )
        
    def initialize_distortions(self):
        self.set_warp()
        self.set_wave()
        self.set_scale()
        self.set_rotation()
        self.set_offset()
        self.set_tint()
        self.make_priorities()
        
    def make_priorities(self):
        self.wav_priority = urand(c.wave)
        self.warp_priority = urand(c.warp)
        self.affine_priority = urand(c.affine)
        self.maxv = max(self.wav_priority,self.warp_priority,self.affine_priority)
        self.minv = min(self.wav_priority,self.warp_priority,self.affine_priority)

    def rotate_values(self, num_distorts=1):
        #not particularly safe to use exec(), but should be fine
        distort_list = ['self.set_' + x + '()' for x in \
                            ['scale','wave','warp','offset','rotation','tint']]
        funcs = random.sample(distort_list,num_distorts)
        for func in funcs:
            exec(func)
        self.make_priorities()
        
    def smart_crop(self, image, to=(180, 180)):
        """will shrink image slightly so that it is free to shift around more"""
        shp = image.shape
        small = cv2.resize(image, to)
        new = np.zeros(shp, np.uint8)
        diff = (shp[0]-to[0], shp[1]-to[1])
        new[diff[0]//2:(shp[0]-diff[0]//2),diff[1]//2:(shp[1]-diff[1]//2)] = small
        return new

    def smart_displacement(self, image, imname=None):
        """will not allow image to be shifted beyond its borders
        imname is used to catch images that throw errors"""
        vsums = np.sum(image, (1,2))
        hsums = np.sum(image, (0,2))
        if not imname:
            imname = 'unknown'
        try:
            hvalid = np.where(hsums > 0.0)
            vvalid = np.where(vsums > 0.0)
            hbounds = (np.min(hvalid), np.max(hvalid))
            vbounds = (np.min(vvalid), np.max(vvalid))
            shp = image.shape
            hrange = (-hbounds[0], shp[0] - hbounds[1])
            vrange = (-vbounds[0], shp[1] - vbounds[1])
            #generate random rolls
            hshift = np.random.randint(hrange[0], hrange[1])
            vshift = np.random.randint(vrange[0], vrange[1])
            #roll it
            image = np.roll(image, hshift, axis=1)
            image = np.roll(image, vshift, axis=0)
        except ValueError:
            print( "%s cannot be properly manipulated..."\
                   "writing to erroneous_image.png" % imname)
            cv2.imwrite('/home/max/workspace/Sketch2/erroneous_image.png', image)
        return image

    def process_image(self, image, smart=False, imname=None):
        """processes image using distortions and some strokes"""
        stroke_first = np.random.random() < c.stroke_priority        
        #handle tint
        for j, tint in enumerate(self.tint):
            if not self.use_color:
                continue
            #will not illuminate 0-alpha tiles
            image[:,:,j] = np.uint8((image[:,:,3] > 0) *
                np.maximum(0,np.minimum(255,
                                        np.uint(image[:,:,j]) + tint
                                        )))
        max_strokes = c.max_strokes
        
        if smart:
            image = self.smart_crop(image)

        stroke_kwargs = copy(c.stroke_kwargs)
        stroke_kwargs['max_num_imps'] = max_strokes
        if stroke_first and max_strokes > 0:
            add_imperfections(image, **c.stroke_kwargs)
        #now do distortions
        if self.wav_priority == self.maxv:
            image = self.wave_distort.process_image(image)
            if self.warp_priority == self.minv:
                if (self.scx + self.scy)/2 > 1:
                    image = self.scale_distort.process_image(image)
                    image = self.rot_distort.process_image(image)
                else:
                    image = self.rot_distort.process_image(image)
                    image = self.scale_distort.process_image(image)
                image = self.warp_distort.process_image(image)
            else:
                image = self.warp_distort.process_image(image)            
                if (self.scx + self.scy)/2 > 1:
                    image = self.scale_distort.process_image(image)
                    image = self.rot_distort.process_image(image)
                else:
                    image = self.rot_distort.process_image(image)
                    image = self.scale_distort.process_image(image)
        elif self.warp_priority == self.maxv:
            image = self.warp_distort.process_image(image)
            if self.wav_priority == self.minv:
                if (self.scx + self.scy)/2 > 1:
                    image = self.scale_distort.process_image(image)
                    image = self.rot_distort.process_image(image)
                else:
                    image = self.rot_distort.process_image(image)
                    image = self.scale_distort.process_image(image)
                image = self.wave_distort.process_image(image)
            else:
                image = self.wave_distort.process_image(image)            
                if (self.scx + self.scy)/2 > 1:
                    image = self.scale_distort.process_image(image)
                    image = self.rot_distort.process_image(image)
                else:
                    image = self.rot_distort.process_image(image)
                    image = self.scale_distort.process_image(image)
        else:
            if (self.scx + self.scy)/2 > 1:
                image = self.scale_distort.process_image(image)
                image = self.rot_distort.process_image(image)
            else:
                image = self.rot_distort.process_image(image)
                image = self.scale_distort.process_image(image)
            if self.wav_priority == self.minv:
                image = self.warp_distort.process_image(image)
                image = self.wave_distort.process_image(image)
            else:
                image = self.wave_distort.process_image(image)
                image = self.warp_distort.process_image(image)
        #displacement
        if not smart:
            image = self.displace.process_image(image)
        else:
            image = self.smart_displacement(image, imname)
        if (not stroke_first) and max_strokes > 0 :
            add_imperfections(image, **stroke_kwargs)
        if np.random.random() < c.flip_chance:
            image = np.fliplr(image)
        return image
