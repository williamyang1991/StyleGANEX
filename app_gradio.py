from __future__ import annotations

import argparse
import pathlib
import torch
import gradio as gr

from webUI.app_task import *
from webUI.styleganex_model import Model


DESCRIPTION = '''
<div align=center>
<h1 style="font-weight: 900; margin-bottom: 7px;">
   Face Manipulation with <a href="https://github.com/williamyang1991/StyleGANEX">StyleGANEX</a>
</h1>
<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. 
<a href="https://huggingface.co/spaces/PKUWilliamYang/StyleGANEX?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>
<p/>
<img style="margin-top: 0em" src="https://raw.githubusercontent.com/williamyang1991/tmpfile/master/imgs/example.jpg" alt="example">
</div>
'''
ARTICLE = r"""
If StyleGANEX is helpful, please help to ⭐ the <a href='https://github.com/williamyang1991/StyleGANEX' target='_blank'>Github Repo</a>. Thanks! 
[![GitHub Stars](https://img.shields.io/github/stars/williamyang1991/StyleGANEX?style=social)](https://github.com/williamyang1991/StyleGANEX)
---
📝 **Citation**
If our work is useful for your research, please consider citing:
```bibtex
@article{yang2023styleganex,
  title = {StyleGANEX: StyleGAN-Based Manipulation Beyond Cropped Aligned Faces},
  author = {Yang, Shuai and Jiang, Liming and Liu, Ziwei and and Loy, Chen Change},
  journal = {arXiv preprint arXiv:2303.06146},
  year={2023},
}
```
📋 **License**
This project is licensed under <a rel="license" href="https://github.com/williamyang1991/VToonify/blob/main/LICENSE.md">S-Lab License 1.0</a>. 
Redistribution and use for non-commercial purposes should follow this license.

📧 **Contact**
If you have any questions, please feel free to reach me out at <b>williamyang@pku.edu.cn</b>.
"""

FOOTER = '<div align=center><img id="visitor-badge" alt="visitor badge" src="https://visitor-badge.laobi.icu/badge?page_id=williamyang1991/styleganex" /></div>'


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('*** Now using %s.'%(device))
    model = Model(device=device)

    torch.hub.download_url_to_file('https://raw.githubusercontent.com/williamyang1991/StyleGANEX/main/data/234_sketch.jpg',
        '234_sketch.jpg')
    torch.hub.download_url_to_file('https://github.com/williamyang1991/StyleGANEX/raw/main/output/ILip77SbmOE_inversion.pt',
        'ILip77SbmOE_inversion.pt')
    torch.hub.download_url_to_file('https://raw.githubusercontent.com/williamyang1991/StyleGANEX/main/data/ILip77SbmOE.png',
        'ILip77SbmOE.png')
    torch.hub.download_url_to_file('https://raw.githubusercontent.com/williamyang1991/StyleGANEX/main/data/ILip77SbmOE_mask.png',
        'ILip77SbmOE_mask.png')
    torch.hub.download_url_to_file('https://raw.githubusercontent.com/williamyang1991/StyleGANEX/main/data/pexels-daniel-xavier-1239291.jpg',
        'pexels-daniel-xavier-1239291.jpg')
    torch.hub.download_url_to_file('https://github.com/williamyang1991/StyleGANEX/raw/main/data/529_2.mp4',
        '529_2.mp4')
    torch.hub.download_url_to_file('https://github.com/williamyang1991/StyleGANEX/raw/main/data/684.mp4',
        '684.mp4')
    torch.hub.download_url_to_file('https://github.com/williamyang1991/StyleGANEX/raw/main/data/pexels-anthony-shkraba-production-8136210.mp4',
        'pexels-anthony-shkraba-production-8136210.mp4')        
    
    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(DESCRIPTION) 
        with gr.Tabs():
            with gr.TabItem('Inversion for Editing'):
                create_demo_inversion(model.process_inversion, allow_optimization=True)          
            with gr.TabItem('Image Face Toonify'):
                create_demo_toonify(model.process_toonify)               
            with gr.TabItem('Video Face Toonify'):
                create_demo_vtoonify(model.process_vtoonify, max_frame_num=5000)                      
            with gr.TabItem('Image Face Editing'):
                create_demo_editing(model.process_editing)  
            with gr.TabItem('Video Face Editing'):
                create_demo_vediting(model.process_vediting, max_frame_num=5000)               
            with gr.TabItem('Sketch2Face'):
                create_demo_s2f(model.process_s2f)   
            with gr.TabItem('Mask2Face'):
                create_demo_m2f(model.process_m2f)  
            with gr.TabItem('SR'):
                create_demo_sr(model.process_sr)  
        gr.Markdown(ARTICLE)
        gr.Markdown(FOOTER)
    
    demo.queue(concurrency_count=1)
    demo.launch(server_port=8088, server_name="0.0.0.0", debug=True)

if __name__ == '__main__':
    main()

