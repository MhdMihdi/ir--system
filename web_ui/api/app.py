from flask import Flask, render_template, request, redirect, url_for, jsonify
from services.dataset_loader import preload_all_datasets
from services.retrieval_service import retrieve_results
from services.query_refinement import QueryRefinement

app = Flask(__name__)

query_refiner = QueryRefinement()
preload_all_datasets()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/search", methods=["GET", "POST"])
def search():
    print("📥 search /")
    if request.method == "POST":
        query = request.form.get("query")
        dataset = request.form.get("dataset")
        model = request.form.get("mode")
        use_vectorstore = request.args.get("use_vectorstore") == "on"
        use_refinement = request.args.get("use_refinement") == "on"

        return redirect(url_for("search", query=query, dataset=dataset, mode=model,
                        use_vectorstore="on" if request.form.get("use_vectorstore") else "off",
                        use_refinement="on" if request.form.get("use_refinement") else "off"))

    
    query = request.args.get("query")
    dataset = request.args.get("dataset")
    model = request.args.get("mode")
    use_vectorstore = request.args.get("use_vectorstore") == "on"
    use_refinement = request.args.get("use_refinement") == "on"

    results = None
    suggestions = None
    if query and dataset and model:
            suggestions = query_refiner.refine_query(query) if use_refinement else None
            results = retrieve_results(query, dataset, model, use_vectorstore=use_vectorstore)

    return render_template(
        "index.html",
        query=query,
        dataset=dataset,
        mode=model,
        results=results,
        suggestions=suggestions
    )

@app.route("/suggest", methods=["GET"])
def suggest():
    """
    واجهة API للحصول على اقتراحات بشكل مباشر
    """
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
        
    suggestions = query_refiner.refine_query(query)
    return jsonify(suggestions)