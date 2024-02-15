import gradio as gr
from cosine_similarity_baseline import generate_top_ten
from cosine_similarity_keyword import generate_top_ten_v2
from cosine_similarity_merged import generate_top_ten_v3, generate_top_ten_v4, generate_top_ten_with_prices


def gradio_gui():
    
    with gr.Blocks() as demo:
        keywords = gr.Textbox(label="Keywords (seperate with space)")
        semantics = gr.Radio(['both', 'pos', 'neg'], label="Do you want to see positive or negative review? (default is both)")
        years = gr.CheckboxGroup(["2000","2001", "2002","2003", "2004", "2005"], label="Year", info="What year of review?")
        outputs = gr.Dataframe(row_count = (10, "dynamic"), col_count=(2, "dynamic"), label="Generated List")
        err_msg = gr.Textbox(label="Error message")
        with gr.Tab("Keyword Curated List"):
            generate_btn = gr.Button("generate")
            generate_btn.click(fn = generate_top_ten_v4, inputs = [keywords,semantics,years], outputs = [outputs,err_msg])
        
        with gr.Tab("Keyword Curated List with prices (slow)"):
            generate_btn = gr.Button("generate")
            generate_btn.click(fn = generate_top_ten_with_prices, inputs = [keywords,semantics,years], outputs = [outputs,err_msg])
            
        with gr.Tab("Baseline Curated List"):
            generate_btn = gr.Button("generate")
            generate_btn.click(fn = generate_top_ten_v3, inputs = [keywords,semantics,years], outputs = [outputs,err_msg])
        
        with gr.Tab("Keyword Cosine Similarity"):
            generate_btn = gr.Button("generate")
            generate_btn.click(fn = generate_top_ten_v2, inputs = [keywords,semantics,years], outputs = [outputs,err_msg])
        
        with gr.Tab("Baseline Cosine Similarity"):
            generate_btn = gr.Button("generate")
            generate_btn.click(fn = generate_top_ten, inputs = [keywords,semantics,years], outputs = [outputs,err_msg])

    demo.queue()
    demo.launch(share=True)
    # demo.launch()
    
    
if __name__ == "__main__":
    gradio_gui()