# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:43:04 2023

@author: Frederic Klein
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
from PIL import Image
import torch
import tqdm
import time

# add path for controlnet extension
path_webui = os.path.dirname(os.path.realpath(__file__))
controlnet_path_script = path_webui + "\extensions\sd-webui-controlnet"
#print(controlnet_path_script)
sys.path.append(controlnet_path_script)

# initialization from webui
from modules import timer
from modules import initialize

startup_timer = timer.startup_timer
startup_timer.record("launcher")
initialize.imports()

# load other webui libraries
from modules.cmd_args import parser
from modules import scripts
from modules import codeformer_model

#codeformer_model.setup_model(dirname="CodeFormer")
from modules.processing import StableDiffusionProcessingImg2Img, process_images
from modules.sd_models import CheckpointInfo, load_model, read_state_dict
from modules import shared
from modules.lang_sam.lang_sam import LangSAM

# load webui extension libraries
from scripts.controlnet import Script as ControlScript
from internal_controlnet.external_code import ControlNetUnit
from modules.processing_scripts import refiner, seed

def scale_image(image, new_width, new_height):
    try:
        rescaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return rescaled
    except IOError:
        print("Cannot resize the image for '%s'" % image)
        raise IOError
        
def get_mask_and_pose(image, pose_array, mask_width=150, mask_height=300):
    # numpy uses different orientation, so change x,y to y,x
    mask_width, mask_height = mask_height, mask_width
    
    # rescale pose image
    pose_image = Image.fromarray(pose_array)
    pose_image = scale_image(pose_image, new_width=mask_height, new_height=mask_width)
    pose = np.array(pose_image)
    #print(pose.shape)
    
    # load model to detect space
    model = LangSAM(sam_type="vit_b")
    image = image.convert("RGB")
    text_prompt = "street"
    
    # detect free pixels
    masks, boxes, phrases, logits = model.predict(image, text_prompt)
    masks = masks.numpy()
    masks = masks[0,:,:]

    # masks contains non-zero value for indices that belong to text_prompt
    # all other values are zero
    indices = np.where(masks)
    indices_copy = np.copy(indices)
    
    print("Searching for a central mask position")
    indices_list = [i for i in range(len(indices[0]))]
    while True:
        try:
            ii = np.random.choice(indices_list, replace=False)
            indices_list.pop(ii)
            xi, yi = indices[0][ii], indices[1][ii]
            # minimum of unbroken road pixels
            if xi > 1/5 * masks.shape[0] and xi < 3/5 * masks.shape[0]:
                if yi+mask_height > 1/5 * masks.shape[1] and yi+mask_height < 4/5 * masks.shape[1]:
                    break
        except ValueError:
            print("Failed to find a central mask position, randomly choosing one.")
            # indices_list is empty
            ii = np.random.choice(indices_copy, replace=False)
            xi, yi = indices[0][ii], indices[1][ii]
    
    # create mask and shift pose
    final_mask = np.zeros_like(masks)
    final_pose = np.zeros(shape=(final_mask.shape[0], final_mask.shape[1], 3),dtype='uint8')
    
    #use the free pixel as the center of the width of the mask
    if int(xi - mask_width/2) >= 0 and xi + mask_width/2 < final_mask.shape[0]:
        xi = int(xi - mask_width/2)
    try:
        if xi+mask_width < final_mask.shape[0]:
            if yi+mask_height < final_mask.shape[1]:
                final_mask[xi:xi+mask_width, yi:yi+mask_height] = 255
                final_pose[xi:xi+mask_width, yi:yi+mask_height,:] = pose[:,:,:3]
                xi2 = xi+mask_width
                yi2 = yi+mask_height
            elif yi-mask_height > 0:
                final_mask[xi:xi+mask_width, yi-mask_height:yi] = 255
                final_pose[xi:xi+mask_width, yi-mask_height:yi,:] = pose[:,:,:3]
                xi2 = xi+mask_width
                yi2 = yi-mask_height
            else:
                raise IndexError
        elif xi-mask_width > 0:
            if yi+mask_height < final_mask.shape[1]:
                final_mask[xi-mask_width:xi, yi:yi+mask_height] = 255
                final_pose[xi-mask_width:xi, yi:yi+mask_height,:] = pose[:,:,:3]
                xi2 = xi-mask_width
                yi2 = yi+mask_height
            elif yi-mask_height > 0:
                final_mask[xi-mask_width:xi, yi-mask_height:yi] = 255
                final_pose[xi-mask_width:xi, yi-mask_height:yi,:] = pose[:,:,:3]
                xi2 = xi-mask_width
                yi2 = yi-mask_height
            else:
                raise IndexError
        else:
            raise IndexError
    except IndexError:
        print("Something went horribly wrong when choosing a mask automatically. Try to decrease the size of the mask, or upload an existing mask and pose.")
    return Image.fromarray(final_mask).convert(mode="L"), final_pose, xi, yi, xi2, yi2

if __name__ == "__main__":
    starting_time = time.time()
    # clean GPU cache
    torch.cuda.empty_cache()
    # load scripts first
    scripts.load_scripts()
    
    args = parser.parse_args()

    # critical errors:
    if args.filepath_model is None:
        raise ValueError
    if args.image_path is None:
        raise ValueError
    if args.prompt is None:
        raise ValueError
    
    # batch_size
    if args.batch_size is None:
        args.batch_size = 1
    args.prompt = args.prompt * args.batch_size
    # images
    image_paths = []
    if args.image_path.endswith(".png"):
        print("Single image input")
        image_paths.append(args.image_path)
    else:
        folder_list = os.listdir(args.image_path)
        
        for fl in folder_list:
            if fl.endswith(".png"):
                image_paths.append(os.path.join(args.image_path, fl))    
        print(f"Folder of {len(image_paths)} images")
    #print(image_paths)

    # stable diffusion model
    checkpoint_info = CheckpointInfo(filename=args.filepath_model)
    checkpoint_info.register()
    state_dict = read_state_dict(checkpoint_info.filename)
    shared.sd_model = load_model(checkpoint_info=checkpoint_info)
    # set low_vram = True
    shared.sd_model.lowvram = True
    #shared.cmd_opts.force_enable_xformers = True
    
    # controlnet model
    args.controlnet_name = args.controlnet_name.replace(".pth", "")
    controlnetscript_object = ControlScript()
    controlnetscript_object.args_from = 10
    controlnetscript_object.args_to = 13
    my_controlnet = ControlNetUnit(model=args.controlnet_name,
                                   low_vram = True,
                                   resize_mode="Crop and Resize",
                                   #processor_res=512,
                                   control_mode="ControlNet is more important")
    
    # refiner script
    scriptrefiner_object = refiner.ScriptRefiner()
    scriptrefiner_object.argsfrom = 1
    scriptrefiner_object.argsto = 4
    
    # seed script
    scriptseed_object = seed.ScriptSeed()
    scriptseed_object.argsfrom = 4
    scriptseed_object.argsfrom = 10
    
    if args.mask_height is None:
        args.mask_height = 300
    if args.mask_width is None:
        args.mask_width = 150

    # loop over images in green
    for ip in tqdm.tqdm(image_paths, colour="green"):
        torch.cuda.empty_cache()
        
        # image
        init_images = []
        with Image.open(ip) as image:
            
            image = scale_image(image, new_width=768, new_height=512)
            image_array = np.asarray(image)
            image_rgb = Image.fromarray(image_array, mode="RGB")#.convert("RGB")
            
            init_images.append(image_rgb)
            # save cropped image
            # image_rgb.save(ip.replace(".png", "_cropped.png"))
        
        # negative prompt
        if args.negative_prompt is None:
            args.negative_prompt = ""
        
        # control_image
        if args.pose_path is None:
            pose_array = np.zeros(shape=(10, 10))
        else:
            pose_array = None
            with Image.open(args.pose_path) as pose:    
                pose_array = np.array(pose)
        
        # mask
        if args.mask_path is not None:
            print("loading the mask.")
            mask = None
            with open(args.mask_path, "r") as mfile:
                coordinates2 = mfile.read()
                coordinates2 = coordinates2.split("\n")
                
                xi, yi = coordinates2[0].split(",")
                xi, yi = int(xi), int(yi)
                xi2, yi2 = coordinates2[1].split(",")
                xi2, yi2 = int(xi2), int(yi2)
                if xi > xi2:
                    xi, xi2 = xi2, xi
                if yi > yi2:
                    yi, yi2 = yi2, yi
            
            mask = np.zeros(shape=(512, 768))
            mask[xi:xi2, yi:yi2] = 255
            
            pose_array2 = np.zeros(shape=(mask.shape[0], mask.shape[1], 3),dtype='uint8')
            pose_array2[xi:xi2, yi:yi2,:] = pose_array[:,:,:3]
            pose_array = pose_array2
            mask = Image.fromarray(mask)
            mask.show()
            Image.fromarray(pose_array, mode="RGB").show()
        else:
            print("Generating new mask and shifting the pose.")
            mask, pose_array, xi, yi, xi2, yi2 = get_mask_and_pose(init_images[0], pose_array, mask_width = args.mask_width, mask_height=args.mask_height)
            with open(args.image_path + "/mask_position.txt", "w") as mask_position_file:
                mask_position_file.write(str(xi))
                mask_position_file.write(", ")
                mask_position_file.write(str(yi))
                mask_position_file.write("\n")
                mask_position_file.write(str(xi2))
                mask_position_file.write(", ")
                mask_position_file.write(str(yi2))
            mask.show()
            pose_image = Image.fromarray(pose_array)

        # seed
        if args.seed is None:
            args.seed = -1
        
        # sampler
        if args.sampler_name is None:
            args.sampler_name = "Euler a"
        
        prompt_styles = []
        n_iter = 1
        steps = 20
        cfg_scale = 7
        width = 512
        height = 512
        mask_blur = 4
        inpainting_fill = 1
        resize_mode = 0
        denoising_strength = 0.75
        image_cfg_scale = 1.5
        inpaint_full_res = 1
        inpaint_full_res_padding = 32
        inpainting_mask_invert = 0
        override_settings = {}
        
        p = StableDiffusionProcessingImg2Img(
            sd_model=shared.sd_model,
            outpath_samples= args.image_path,
            outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_img2img_grids,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            styles=prompt_styles,
            sampler_name=args.sampler_name,
            batch_size=1,
            n_iter=n_iter,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            init_images=init_images,
            mask=mask,
            mask_blur=mask_blur,
            inpainting_fill=inpainting_fill,
            resize_mode=resize_mode,
            denoising_strength=denoising_strength,
            image_cfg_scale=image_cfg_scale,
            inpaint_full_res_padding=inpaint_full_res_padding,
            inpainting_mask_invert=inpainting_mask_invert,
            override_settings=override_settings,
            inpaint_full_res=inpaint_full_res,
        )
        # set arguments the way stable diffusion webui handles them from GUI
        # your best option to find these is to run the GUI and print them out in img2img.py img2img()
        p_args = (0, False, '', 0.8, 42, False, -1, 0, 0, 0, my_controlnet, controlnetscript_object.get_default_ui_unit(), controlnetscript_object.get_default_ui_unit(), '* `CFG Scale` should be 2 or lower.', True, True, '', '', True, 50, True, 1, 0, False, 4, 0.5, 'Linear', 'None', '<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>', 128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'], False, False, 'positive', 'comma', 0, False, False, '', '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>', 64, 0, 2, 1, '', [], 0, '', [], 0, '', [], True, False, False, False, 0, False, None, None, False, None, None, False, None, None, False, 50)
        p.scripts = scripts.scripts_img2img
        p.script_args = p_args
        p.controlnet_input_image = pose_array
        p.seed = args.seed
        # face restoration
        p.restore_faces = False
        #face_restorers = []
        #for fr in shared.face_restorers:
        #    if isinstance(fr, str):
        #        continue
        #    face_restorers.append(fr)
        #shared.face_restorers = face_restorers
        p.is_using_inpainting_conditioning=False
        p.paste_to = None
        p.seed = args.seed
        p.seed_enable_extras = True
        p.tiling = None
        p.restore_faces = None
        p.do_not_save_samples = False
        p.do_not_save_grid = False
        p.overlay_images = None
        p.eta = None
        p.do_not_reload_embeddings = False
        p.ddim_discretize = None
        p.s_min_uncond = 0.0
        p.s_churn = 0.0
        p.s_tmax = "inf"
        p.s_tmin = 0.0
        p.s_noise = 1.0
        p.override_settings_restore_afterwards = True
        p.sampler_index = None
        p.refiner_checkpoint = None
        p.refiner_switch_at = None
        p.disable_extra_networks = False
        p.comments = {}
        p.mask_blur_x = 4
        p.mask_blur_y = 4
        p.initial_noise_multiplier = 1.0
        p.latent_mask = None
        p.sampler_noise_scheduler_override = None
        p.refiner_checkpoint_info = None
        p.cached_uc = [None, None]
        p.cached_c = [None, None]
        p.user = None
    
        
        p.scripts.alwayson_scripts = [scriptrefiner_object, scriptseed_object, controlnetscript_object]
        
        default_seed = None
        batch_index = 0
        for prompt in tqdm.tqdm(args.prompt, colour="blue"):
            p.prompt = prompt
            title = prompt.replace(",", "-")
            p.outpath_samples= args.image_path + "/" + title
            if default_seed is None:
                if p.seed != -1:
                    default_seed = p.seed
            else:
                p.seed = default_seed + batch_index
            print("p.seed", p.seed)
            batch_index += 1
            batch_index = batch_index % args.batch_size
            torch.cuda.empty_cache()
            # start the process
            p_args = (0, False, '', 0.8, 42, False, -1, 0, 0, 0, my_controlnet, controlnetscript_object.get_default_ui_unit(), controlnetscript_object.get_default_ui_unit(), '* `CFG Scale` should be 2 or lower.', True, True, '', '', True, 50, True, 1, 0, False, 4, 0.5, 'Linear', 'None', '<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>', 128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'], False, False, 'positive', 'comma', 0, False, False, '', '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>', 64, 0, 2, 1, '', [], 0, '', [], 0, '', [], True, False, False, False, 0, False, None, None, False, None, None, False, None, None, False, 50)
            #result = p.scripts.run(p, *p_args)
            result = p.scripts.run(p, *p.script_args)
            
            if result is None:
                result = process_images(p)
    time = time.time() - starting_time
    print(time)
