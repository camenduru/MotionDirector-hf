import gradio as gr
import os
from demo.motiondirector import MotionDirector

from huggingface_hub import snapshot_download

snapshot_download(repo_id="cerspense/zeroscope_v2_576w", local_dir="./zeroscope_v2_576w/")
snapshot_download(repo_id="ruizhaocv/MotionDirector", local_dir="./MotionDirector_pretrained")

is_spaces = True if "SPACE_ID" in os.environ else False
true_for_shared_ui = False  # This will be true only if you are in a shared UI
if (is_spaces):
    true_for_shared_ui = True if "ruizhaocv/MotionDirector" in os.environ['SPACE_ID'] else False



runner = MotionDirector()


def motiondirector(model_select, text_pormpt, neg_text_pormpt, random_seed=1, steps=25, guidance_scale=7.5, baseline_select=False):
    return runner(model_select, text_pormpt, neg_text_pormpt, int(random_seed) if random_seed != "" else 1, int(steps), float(guidance_scale), baseline_select)


with gr.Blocks() as demo:
    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
        <a href="https://github.com/showlab/MotionDirector" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
        </a>
        <div>
            <h1 >MotionDirector: Motion Customization of Text-to-Video Diffusion Models</h1>
            <h5 style="margin: 0;">More MotionDirectors are on the way. Stay tuned 🔥!</h5>
             <h5 style="margin: 0;"> If you like our project, please give us a star ✨ on Github for the latest update.</h5>
            </br>
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;>
                <a href="https://arxiv.org/abs/2310.08465"></a>
                <a href="https://arxiv.org/abs/2310.08465"><img src="https://img.shields.io/badge/arXiv-2310.08465-b31b1b.svg"></a>&nbsp;&nbsp;
                <a href="https://showlab.github.io/MotionDirector"><img src="https://img.shields.io/badge/Project_Page-MotionDirector-green"></a>&nbsp;&nbsp;
                <a href="https://github.com/showlab/MotionDirector"><img src="https://img.shields.io/badge/Github-Code-blue"></a>&nbsp;&nbsp;
            </div>
        </div>
        </div>
        """)
    with gr.Row():
        generated_video_baseline = gr.Video(format="mp4", label="Video Generated by base model (ZeroScope with same seed)")
        generated_video = gr.Video(format="mp4", label="Video Generated by MotionDirector")

        with gr.Column():
            baseline_select = gr.Checkbox(label="Compare with baseline (ZeroScope with same seed)", info="Run baseline? Note: Inference time will be doubled.")
            random_seed = gr.Textbox(label="Random seed", value=1, info="default: 1")
            sampling_steps = gr.Textbox(label="Sampling steps", value=30, info="default: 30")
            guidance_scale = gr.Textbox(label="Guidance scale", value=12, info="default: 12")

    with gr.Row():
        model_select = gr.Dropdown(
            ["1-1: [Cinematic Shots] -- Zoom Out",
             "1-2: [Cinematic Shots] -- Zoom In",
             "1-3: [Cinematic Shots] -- Zoom Out",
             "1-3: [Cinematic Shots] -- Dolly Zoom (Hitchcockian Zoom) 1",
             "1-4: [Cinematic Shots] -- Dolly Zoom (Hitchcockian Zoom) 2",
             "1-5: [Cinematic Shots] -- Follow",
             "1-6: [Cinematic Shots] -- Reverse Follow",
             "1-7: [Cinematic Shots] -- Chest Transition",
             "1-8: [Cinematic Shots] -- Mini Jib Reveal",
             "1-9: [Cinematic Shots] -- Orbit",
             "1-10: [Cinematic Shots] -- Pull Back",
             "2-1: [Object Trajectory] -- Right to Left",
             "2-2: [Object Trajectory] -- Left to Right",
             "3-1: [Sports Concepts] -- Riding Bicycle",
             "3-2: [Sports Concepts] -- Riding Horse",
             "3-3: [Sports Concepts] -- Lifting Weights",
             "3-4: [Sports Concepts] -- Playing Golf",
             "3-5: [Sports Concepts] -- Skateboarding",
             ],
            label="MotionDirector",
            info="Which MotionDirector would you like to use!"
        )

        text_pormpt = gr.Textbox(label="Text Prompt", value='', placeholder="Input your text prompt here!")
        neg_text_pormpt = gr.Textbox(label="Negative Text Prompt", value='', placeholder="default: None")

    submit = gr.Button("Generate")

    # when the `submit` button is clicked
    submit.click(
        motiondirector,
        [model_select, text_pormpt, neg_text_pormpt, random_seed, sampling_steps, guidance_scale, baseline_select],
        [generated_video, generated_video_baseline]
    )

    # Examples
    gr.Markdown("## Examples")
    gr.Examples(
        fn=motiondirector,
        examples=[
            ["1-1: [Cinematic Shots] -- Zoom Out", "A spaceman standing on the moon captured with a zoom out.",
             8323920],
            ["1-2: [Cinematic Shots] -- Zoom In", "A polar bear standing at the top of a snowy mountain captured with a zoom in.", 7938587],
            ["1-3: [Cinematic Shots] -- Dolly Zoom (Hitchcockian Zoom) 1", "A panda standing in front of an ancient Chinese temple captured with a dolly zoom.", 8238823],
            ["1-4: [Cinematic Shots] -- Dolly Zoom (Hitchcockian Zoom) 2", "A lion sitting on top of a cliff captured with a dolly zoom.", 1675932],
            ["1-5: [Cinematic Shots] -- Follow", "A fireman is walking through fire captured with a follow cinematic shot.", 2927089],
            ["1-6: [Cinematic Shots] -- Reverse Follow", "A fireman is walking through fire captured with a reverse follow cinematic shot.", 9759630],
            ["1-7: [Cinematic Shots] -- Chest Transition", "An ancient Roman soldier walks through the crowd on the street captured with a chest transition cinematic shot.", 3982271],
            ["1-8: [Cinematic Shots] -- Mini Jib Reveal",
             "A British Redcoat soldier is walking through the mountains captured with a mini jib reveal cinematic shot.",
             566917],
            ["1-9: [Cinematic Shots] -- Orbit", "A spaceman on the moon captured with an orbit cinematic shot.", 5899496],
            ["1-10: [Cinematic Shots] -- Pull Back", "A spaceman on the moon looking at a lunar rover captured with a pull back cinematic shot.",
             5585865],
            ["2-1: [Object Trajectory] -- Right to Left", "A tank is running on the moon.", 2047046],
            ["2-2: [Object Trajectory] -- Left to Right", "A tiger is running in the forest.", 3463673],
            ["3-1: [Sports Concepts] -- Riding Bicycle", "An astronaut is riding a bicycle past the pyramids Mars 4K high quailty highly detailed.", 4422954],
            ["3-2: [Sports Concepts] -- Riding Horse", "A man riding an elephant through the jungle.", 6230765],
            ["3-3: [Sports Concepts] -- Lifting Weights", "A panda is lifting weights in a garden.", 1699276],
            ["3-4: [Sports Concepts] -- Playing Golf", "A monkey is playing golf on a field full of flowers.", 4156856],
            ["3-5: [Sports Concepts] -- Skateboarding", "An astronaut is skateboarding on Mars.", 6615212],
        ],
        inputs=[model_select, text_pormpt, random_seed],
        outputs=generated_video,
    )

demo.queue(max_size=15)
demo.launch(share=False)