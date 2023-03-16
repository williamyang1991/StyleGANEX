from __future__ import annotations
from huggingface_hub import hf_hub_download
import numpy as np
import gradio as gr


def create_demo_sr(process):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Face Super Resolution')
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type='filepath')
                model_type = gr.Radio(label='Model Type', choices=['SR for 32x','SR for 4x-48x'], value='SR for 32x')
                resize_scale = gr.Slider(label='Resize Scale',
                                            minimum=4,
                                            maximum=48,
                                            value=32,
                                            step=4)                                
                run_button = gr.Button(label='Run')
                gr.Examples(
                    examples =[['pexels-daniel-xavier-1239291.jpg', 'SR for 32x', 32],
                               ['ILip77SbmOE.png', 'SR for 32x', 32],
                               ['ILip77SbmOE.png', 'SR for 4x-48x', 48],
                              ],
                    inputs = [input_image, model_type, resize_scale],
                )                
            with gr.Column():
                #lrinput = gr.Image(label='Low-resolution input',type='numpy', interactive=False)
                #result = gr.Image(label='Output',type='numpy', interactive=False)
                result = gr.Gallery(label='LR input and Output',
                                    elem_id='gallery').style(grid=2,
                                                             height='auto')    

        inputs = [
            input_image,
            resize_scale,
            model_type,
        ]
        run_button.click(fn=process,
                         inputs=inputs,
                         outputs=[result],
                         api_name='sr')
    return demo
    
def create_demo_s2f(process):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Sketch-to-Face Translation')
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type='filepath')
                gr.Markdown("""Note: Input will be cropped if larger than 512x512.""")
                seed = gr.Slider(label='Seed for appearance',
                                    minimum=0,
                                    maximum=2147483647,
                                    step=1,
                                    randomize=True)
                #input_info = gr.Textbox(label='Process Information', interactive=False, value='n.a.')
                run_button = gr.Button(label='Run')
                gr.Examples(
                    examples =[['234_sketch.jpg', 1024]],
                    inputs = [input_image, seed],
                )                
            with gr.Column():
                result = gr.Image(label='Output',type='numpy', interactive=False) 

        inputs = [
            input_image, seed
        ]
        run_button.click(fn=process,
                         inputs=inputs,
                         outputs=[result],
                         api_name='s2f')
    return demo


def create_demo_m2f(process):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Mask-to-Face Translation')
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type='filepath') 
                input_type = gr.Radio(label='Input Type', choices=['color image','parsing mask'], value='color image')
                seed = gr.Slider(label='Seed for appearance',
                                    minimum=0,
                                    maximum=2147483647,
                                    step=1,
                                    randomize=True)
                #input_info = gr.Textbox(label='Process Information', interactive=False, value='n.a.')
                run_button = gr.Button(label='Run')
                gr.Examples(
                    examples =[['ILip77SbmOE.png', 'color image', 4], ['ILip77SbmOE_mask.png', 'parsing mask', 4]],
                    inputs = [input_image, input_type, seed],
                )                
            with gr.Column():
                #vizmask = gr.Image(label='Visualized mask',type='numpy', interactive=False)
                #result = gr.Image(label='Output',type='numpy', interactive=False) 
                result = gr.Gallery(label='Visualized mask and Output',
                                    elem_id='gallery').style(grid=2,
                                                             height='auto')                

        inputs = [
            input_image, input_type, seed
        ]
        run_button.click(fn=process,
                         inputs=inputs,
                         outputs=[result],
                         api_name='m2f')
    return demo

def create_demo_editing(process):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Video Face Editing (for image input)')
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type='filepath') 
                model_type = gr.Radio(label='Editing Type', choices=['reduce age','light hair color'], value='color image')
                scale_factor = gr.Slider(label='editing degree (-2~2)',
                                    minimum=-2,
                                    maximum=2,
                                    value=1,
                                    step=0.1)
                #input_info = gr.Textbox(label='Process Information', interactive=False, value='n.a.')
                run_button = gr.Button(label='Run')
                gr.Examples(
                    examples =[['ILip77SbmOE.png', 'reduce age', -2], 
                               ['ILip77SbmOE.png', 'light hair color', 1]],
                    inputs = [input_image, model_type, scale_factor],
                )                
            with gr.Column():
                result = gr.Image(label='Output',type='numpy', interactive=False) 

        inputs = [
            input_image, scale_factor, model_type
        ]
        run_button.click(fn=process,
                         inputs=inputs,
                         outputs=[result],
                         api_name='editing')
    return demo

def create_demo_toonify(process):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Video Face Toonification (for image input)')
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type='filepath') 
                style_type = gr.Radio(label='Style Type', choices=['Pixar','Cartoon','Arcane'], value='Pixar')                
                #input_info = gr.Textbox(label='Process Information', interactive=False, value='n.a.')
                run_button = gr.Button(label='Run')
                gr.Examples(
                    examples =[['ILip77SbmOE.png', 'Pixar'], ['ILip77SbmOE.png', 'Cartoon'], ['ILip77SbmOE.png', 'Arcane']],
                    inputs = [input_image, style_type],
                )                
            with gr.Column():
                result = gr.Image(label='Output',type='numpy', interactive=False) 

        inputs = [
            input_image, style_type
        ]
        run_button.click(fn=process,
                         inputs=inputs,
                         outputs=[result],
                         api_name='toonify')
    return demo


def create_demo_vediting(process, max_frame_num = 4):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Video Face Editing (for video input)')
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(source='upload', mirror_webcam=False, type='filepath') 
                model_type = gr.Radio(label='Editing Type', choices=['reduce age','light hair color'], value='color image')
                scale_factor = gr.Slider(label='editing degree (-2~2)',
                                    minimum=-2,
                                    maximum=2,
                                    value=1,
                                    step=0.1)
                info = ''
                if max_frame_num < 100:
                    info =  '(full video editing is not allowed so as not to slow down the demo, \
                            but you can duplicate the Space to modify the number limit to a large value)'
                frame_num = gr.Slider(label='Number of frames to edit' + info,
                                    minimum=1,
                                    maximum=max_frame_num,
                                    value=4,
                                    step=1)   
                #input_info = gr.Textbox(label='Process Information', interactive=False, value='n.a.')
                run_button = gr.Button(label='Run')
                gr.Examples(
                    examples =[['684.mp4', 'reduce age', 1.5, 2], 
                               ['684.mp4', 'light hair color', 0.7, 2]],
                    inputs = [input_video, model_type, scale_factor],
                )                
            with gr.Column():
                viz_result = gr.Gallery(label='Several edited frames', elem_id='gallery').style(grid=2, height='auto')  
                result = gr.Video(label='Output', type='mp4', interactive=False) 

        inputs = [
            input_video, scale_factor, model_type, frame_num
        ]
        run_button.click(fn=process,
                         inputs=inputs,
                         outputs=[viz_result, result],
                         api_name='vediting')
    return demo

def create_demo_vtoonify(process, max_frame_num = 4):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Video Face Toonification (for video input)')
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(source='upload', mirror_webcam=False, type='filepath') 
                style_type = gr.Radio(label='Style Type', choices=['Pixar','Cartoon','Arcane'], value='Pixar')
                info = ''
                if max_frame_num < 100:
                    info =  '(full video toonify is not allowed so as not to slow down the demo, \
                            but you can duplicate the Space to modify the number limit from 4 to a large value)'
                frame_num = gr.Slider(label='Number of frames to toonify' + info,
                                    minimum=1,
                                    maximum=max_frame_num,
                                    value=4,
                                    step=1)            
                #input_info = gr.Textbox(label='Process Information', interactive=False, value='n.a.')
                run_button = gr.Button(label='Run')
                gr.Examples(
                    examples =[['529_2.mp4', 'Arcane'],
                               ['pexels-anthony-shkraba-production-8136210.mp4', 'Pixar'], 
                               ['684.mp4', 'Cartoon']],
                    inputs = [input_video, style_type],
                )                
            with gr.Column():
                viz_result = gr.Gallery(label='Several toonified frames', elem_id='gallery').style(grid=2, height='auto')  
                result = gr.Video(label='Output', type='mp4', interactive=False) 

        inputs = [
            input_video, style_type, frame_num
        ]
        run_button.click(fn=process,
                         inputs=inputs,
                         outputs=[viz_result, result],
                         api_name='vtoonify')
    return demo

def create_demo_inversion(process, allow_optimization=False):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## StyleGANEX Inversion for Editing')
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type='filepath') 
                info = ''
                if allow_optimization == False:
                    info = ' (latent optimization is not allowed so as not to slow down the demo, \
                    but you can duplicate the Space to modify the option or directly upload an optimized latent file. \
                    The file can be computed by inversion.py from the github page or colab)'
                optimize = gr.Radio(label='Whether optimize latent' + info, choices=['No optimization','Latent optimization'], 
                                    value='No optimization', interactive=allow_optimization)
                input_latent = gr.File(label='Optimized latent code (optional)', file_types=[".pt"])
                editing_options = gr.Dropdown(['None', 'Style Mixing', 
                                               'Attribute Editing: smile', 
                                               'Attribute Editing: open_eye', 
                                               'Attribute Editing: open_mouth', 
                                               'Attribute Editing: pose', 
                                               'Attribute Editing: reduce_age', 
                                               'Attribute Editing: glasses', 
                                               'Attribute Editing: light_hair_color', 
                                               'Attribute Editing: slender', 
                                               'Domain Transfer: disney_princess',
                                               'Domain Transfer: vintage_comics',
                                               'Domain Transfer: pixar',
                                               'Domain Transfer: edvard_munch',
                                               'Domain Transfer: modigliani',
                                              ], 
                                              label="editing options (based on StyleGAN-NADA, InterFaceGAN, LowRankGAN)",
                                              value='None')
                scale_factor = gr.Slider(label='editing degree (-2~2) for Attribute Editing',
                                    minimum=-2,
                                    maximum=2,
                                    value=2,
                                    step=0.1)
                seed = gr.Slider(label='Appearance Seed for Style Mixing',
                                    minimum=0,
                                    maximum=2147483647,
                                    step=1,
                                    randomize=True)
                #input_info = gr.Textbox(label='Process Information', interactive=False, value='n.a.')
                run_button = gr.Button(label='Run')
                gr.Examples(
                    examples =[['ILip77SbmOE.png', 'ILip77SbmOE_inversion.pt', 'Domain Transfer: vintage_comics'],
                               ['ILip77SbmOE.png', 'ILip77SbmOE_inversion.pt', 'Attribute Editing: smile'],
                               ['ILip77SbmOE.png', 'ILip77SbmOE_inversion.pt', 'Style Mixing'],
                              ], 
                    inputs = [input_image, input_latent, editing_options],
                )                
            with gr.Column():
                result = gr.Image(label='Inversion output',type='numpy', interactive=False) 
                editing_result = gr.Image(label='Editing output',type='numpy', interactive=False) 

        inputs = [
            input_image, optimize, input_latent, editing_options, scale_factor, seed
        ]
        run_button.click(fn=process,
                         inputs=inputs,
                         outputs=[result, editing_result],
                         api_name='inversion')
    return demo
