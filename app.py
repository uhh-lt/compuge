import gradio as gr
import pandas as pd
import requests

backend_url = "https://cam-v2-compuge-db.ltdemos.informatik.uni-hamburg.de/api"

# UI Root
with gr.Blocks() as demo:
    gr.Markdown("## CompUGE-Bench: Comparative Understanding and Generation Evaluation Benchmarks")

    # Main Tabs
    with gr.Tab("Leaderboards"):
        gr.Markdown("### Leaderboards")

        # QC Tab
        with gr.Tab("Question Classification"):
            gr.Markdown("### Question Classification Leaderboard")
            cqi_leaderboard = pd.DataFrame(requests.get(backend_url + "/leaderboard/QI").json())
            cqi_leaderboard.drop(columns=["task"], inplace=True)
            cqi_leaderboard_component = gr.components.Dataframe(cqi_leaderboard)

        # OAI Tab
        with gr.Tab("Object and Aspect Identification"):
            gr.Markdown("### Object & Aspect Identification Leaderboard")
            gr.Markdown("The OAI leaderboard will be opened soon!")

        # SC Tab
        with gr.Tab("Stance Classification"):
            gr.Markdown("### Stance Clasification Leaderboard")
            gr.Markdown("The SC leaderboard will be opened soon!")

        # SG Tab
        with gr.Tab("Summary Generation"):
            gr.Markdown("### Summary Generation Leaderboard")
            gr.Markdown("The Summary Generation leaderboard will be opened soon!")

    # Model Submissions Tab
    with gr.Tab("Model Submissions"):
        gr.Markdown("### Submission")
        gr.Markdown("The submission will be opened soon!")

    # About Tab
    with gr.Tab("About"):
        gr.Markdown("### About")
        gr.Markdown("CompUGE-Bench is a benchmark for comparative understanding and generation evaluation.")

    # Contact Tab
    with gr.Tab("Contact"):
        gr.Markdown("### Contact")
        gr.Markdown("For any questions, please contact us at ahmad.shallouf@uni-hamburg.de")
# Launch public demo
demo.launch()
