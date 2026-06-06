import gradio as gr
from engine import MovieRecommenderEngine

# Initialize the recommendation engine on app startup
try:
    engine = MovieRecommenderEngine()
    user_ids = engine.get_user_ids()
    user_dataset_samples = [[uid] for uid in user_ids]
    rep_movies = engine.get_representative_movies()
except Exception as e:
    import traceback
    print("="*60)
    print(f"CRITICAL ENGINE ERROR: {e}")
    traceback.print_exc()
    print("="*60)
    user_ids = []
    user_dataset_samples = []
    rep_movies = []

def get_recommendations(evt: gr.SelectData):
    if not user_ids:
        return "<p style='color: red; padding: 15px;'>❌ Engine initialization failed.</p>"
    try:
        user_id = int(evt.value[0])
        recs = engine.recommend(user_id, n=12) 
        history_recs = engine.get_top_rated_by_user(user_id, n=5)
        
        html_content = f"<h3 style='color: white; font-family: sans-serif;'>🎬 Top Predictions for User {user_id}</h3><div class='netflix-grid'>"
        for idx, item in enumerate(recs, 1):
            html_content += f"""
            <div class="netflix-card">
                <img src="{item['poster']}" class="card-img" />
                <div class="card-overlay">
                    <div class="card-rank">#{idx}</div>
                    <div class="card-title">{item['title']}</div>
                    <div class="card-genres">{item['genres']}</div>
                </div>
            </div>"""
        html_content += "</div>"
        
        html_content += "<hr style='border: 0; border-top: 1px solid #333; margin: 30px 0;'>"
        html_content += f"<h3 style='color: white; font-family: sans-serif;'>⭐ User {user_id}'s Top Highest Rated Movies</h3><div class='netflix-grid'>"
        for idx, item in enumerate(history_recs, 1):
            html_content += f"""
            <div class="netflix-card">
                <img src="{item['poster']}" class="card-img" />
                <div class="card-overlay">
                    <div class="card-rating">★ {item['rating']}</div>
                    <div class="card-title">{item['title']}</div>
                    <div class="card-genres">{item['genres']}</div>
                </div>
            </div>"""
        html_content += "</div>"
        return html_content
    except Exception as e:
        return f"<p style='color: red;'>An error occurred: {e}</p>"

def open_modal():
    return gr.update(visible=True)

def close_modal():
    return gr.update(visible=False), []

def process_selections_and_recommend(selected_movies):
    if not selected_movies:
        return (
            gr.update(visible=False), 
            "<div style='padding: 25px; text-align: center; color: #777;'><h3>⚠️ No movies selected. Please choose at least one movie.</h3></div>"
        )
        
    try:
        recs = engine.recommend_for_new_user(selected_movies, n=12)
        html_content = f"<h3>✨ Tailored Recommendations Based On Your Taste</h3>"
        html_content += f"<p style='color: #aaa; margin-bottom: 20px;'>Because you enjoy: <i>{', '.join(selected_movies[:4])}...</i></p>"
        html_content += "<div class='netflix-grid'>"
        for idx, item in enumerate(recs, 1):
            html_content += f"""
            <div class="netflix-card">
                <img src="{item['poster']}" class="card-img" />
                <div class="card-overlay">
                    <div class="card-rank">#{idx}</div>
                    <div class="card-title">{item['title']}</div>
                    <div class="card-genres">{item['genres']}</div>
                </div>
            </div>"""
        html_content += "</div>"
        return gr.update(visible=False), html_content
    except Exception as e:
        return gr.update(visible=False), f"<p style='color: red;'>Error generating recommendations: {e}</p>"

def toggle_explanation_view():
    return gr.update(visible=True)

def hide_explanation_view():
    return gr.update(visible=False)

custom_css = """
#scroll-container { max-height: 800px; overflow-y: auto !important; }
.netflix-grid { display: flex; flex-wrap: wrap; gap: 14px; padding: 5px 0; width: 100%; }
.netflix-card { position: relative; flex: 0 0 calc(16.66% - 14px); min-width: 120px; height: 220px; border-radius: 6px; overflow: hidden; background-color: #222; cursor: pointer; transition: transform 0.2s ease; }
.netflix-card:hover { transform: scale(1.05); }
.card-img { width: 100%; height: 100%; object-fit: cover; }
.card-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: linear-gradient(to top, rgba(0,0,0,0.95) 40%, transparent 100%); display: flex; flex-direction: column; justify-content: flex-end; padding: 10px; box-sizing: border-box; opacity: 0; transition: opacity 0.2s ease; }
.netflix-card:hover .card-overlay { opacity: 1; }
.card-title { color: white; font-size: 11px; font-weight: 700; line-height: 1.2; font-family: sans-serif; }
.card-genres { color: #aaa; font-size: 9px; font-family: sans-serif; }
.card-rank { position: absolute; top: 8px; left: 8px; background: #e50914; color: white; font-weight: bold; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-family: sans-serif; }
.card-rating { position: absolute; top: 8px; left: 8px; background: #d4af37; color: #111; font-weight: bold; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-family: sans-serif; }

#modal_window_wrapper, #explanation_window_wrapper {
    background: #181818;
    border-radius: 12px;
    border: 1px solid #333;
    padding: 20px;
    margin-bottom: 25px;
}
.right-aligned-link {
    text-align: right;
}
.right-aligned-link button {
    background: none !important;
    border: none !important;
    color: #e50914 !important;
    text-decoration: underline !important;
    cursor: pointer !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}
"""

with gr.Blocks(title="MovieLens SVD recommendations", css=custom_css, theme=gr.themes.Soft()) as demo:
    
    # Header Layout Row with Context Link Positioned Right
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("# 🎬 MovieLens SVD recommendations")
        with gr.Column(scale=1, elem_classes="right-aligned-link"):
            why_built_btn = gr.Button("How did I build this?")

    # TECHNICAL ARCHITECTURE DRAWER PANEL
    with gr.Column(visible=False, elem_id="explanation_window_wrapper") as explanation_container:
        gr.Markdown("## 🛠️ System Design & Architectural Engineering")
        
        gr.Markdown(
            "### 1. Replicating Early Netflix (SVD on Static Profiles)\n"
            "This application mimics **early Netflix's historical production philosophy** by separating processing vectors based on data lifecycle rules. "
            "For the predefined user base loaded via the sidebar dataset, recommendations are generated using **Truncated Singular Value Decomposition (SVD)**.\n\n"
            "By running an offline mathematical factorization ($R \\approx U \\Sigma V^T$) across historical interaction rows, the matrix structures are compressed into dense, "
            "low-rank latent embeddings. Because the data catalog is static, serving an existing profile requires nothing more than a highly efficient vector dot product lookup. "
            "This replicates the lightning-fast, predictable retrieval latency early streaming architectures relied on."
        )
        
        gr.Markdown(
            "### 2. The Real-Time Constraint (Why SVD is Unfeasible for Interactive Selection)\n"
            "While SVD excels at querying cold, static data, **it completely breaks down when an app requires interactive, real-time user inputs**. "
            "SVD is mathematically rigid. When a user dynamically checks movie boxes in the interface, they act as an entirely new entity absent from the trained coordinate space. "
            "Mapping a fresh interaction profile into a global SVD latent space requires an extensive, full-matrix retraining cycle. Running this heavy computation synchronously "
            "on a button click is completely unfeasible for a modern web application, as it would lock up the server and destroy user experience."
        )
        
        gr.Markdown(
            "### 3. The Interactive Solution (Neighborhood Co-occurrence Heuristic)\n"
            "To support dynamic multi-select features without server retraining lag, the ad-hoc pipeline seamlessly routes custom inputs to a **Neighborhood-Based Co-occurrence Heuristic**.\n\n"
            "Instead of projecting real-time updates through abstract matrix mathematical constraints, this memory-based fallback performs instant, lightweight statistical matching. "
            "It queries the database to find standard reference peers who highly rated the exact films you just checked, aggregates their *other* favorite movies, and calculates their joint frequencies. "
            "This hybrid approach delivers instant, interactive discovery with sub-second latencies."
        )
        
        with gr.Row():
            gr.Markdown("") # Spacer
            close_exp_btn = gr.Button("Hide Technical Specifications", variant="secondary")

    # CUSTOM CHOOSE MOVIE OVERLAY PANEL
    with gr.Column(visible=False, elem_id="modal_window_wrapper") as modal_container:
        gr.Markdown("### 🍿 Pick Movies You Love")
        gr.Markdown("Select as many options as you like, then press **Generate Dashboard** below:")
        
        movie_choices = [m['title'] for m in rep_movies] if rep_movies else []
        taste_selector = gr.CheckboxGroup(
            choices=movie_choices,
            label="Available Catalog Options",
            interactive=True
        )
        
        with gr.Row():
            cancel_btn = gr.Button("❌ Cancel & Close", variant="stop")
            submit_btn = gr.Button("🚀 Generate Dashboard", variant="primary")

    with gr.Row():
        with gr.Column(scale=1, elem_id="scroll-container", variant="panel"):
            custom_rec_btn = gr.Button("🍿 Get your movie recs (non-SVD method)", variant="primary")
            gr.Markdown("### 👤 Dataset User IDs")
            user_selector = gr.Dataset(
                components=[gr.Textbox(visible=False)], 
                headers=["User ID List"],
                samples=user_dataset_samples,
                type="values",
                samples_per_page=75
            )
            
        with gr.Column(scale=4):
            output_html = gr.HTML(
                value="""
                <div style='padding: 25px; border: 2px dashed #333; border-radius: 8px; text-align: center; color: #777; font-family: sans-serif;'>
                    <h3>🔍 Profile Dashboard Idle</h3>
                    <p>Select a user profile on the left or tap 'Get your movie recs (non-SVD method)' to deploy your custom choice matrices.</p>
                </div>"""
            )

    # Event Wire Bindings
    why_built_btn.click(fn=toggle_explanation_view, inputs=None, outputs=explanation_container)
    close_exp_btn.click(fn=hide_explanation_view, inputs=None, outputs=explanation_container)
    
    custom_rec_btn.click(fn=open_modal, inputs=None, outputs=modal_container)
    cancel_btn.click(fn=close_modal, inputs=None, outputs=[modal_container, taste_selector])
    submit_btn.click(
        fn=process_selections_and_recommend, 
        inputs=taste_selector, 
        outputs=[modal_container, output_html]
    )
    user_selector.select(fn=get_recommendations, inputs=None, outputs=output_html)

if __name__ == "__main__":
    demo.launch()