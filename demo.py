import gradio as gr

with gr.Blocks(title="Text Summarization") as demo:

    content_box = gr.Textbox(label="Content")
    models_box = gr.CheckboxGroup(["BARTPho", "LSTM1", "LSTM2", "LSTM3"], label="Model")
    submit_btn = gr.Button("Submit")

    # with gr.Column(visible=False) as output_col:
    model1 = gr.Textbox(label="BARTPho", visible=False)
    model2 = gr.Textbox(label="LSTM1", visible=False)
    model3 = gr.Textbox(label="LSTM2", visible=False)
    model4 = gr.Textbox(label="LSTM3", visible=False)

    def submit(content, models_box):
        return {
            model1: gr.update(visible="BARTPho" in models_box, value="Model 1"),
            model2: gr.update(visible="LSTM1" in models_box, value="Model 2"),
            model3: gr.update(visible="LSTM2" in models_box, value="Model 3"),
            model4: gr.update(visible="LSTM3" in models_box, value="Model 4"),
        }

    submit_btn.click(
        submit,
        [content_box, models_box],
        [model1, model2, model3, model4],
    )

demo.launch()