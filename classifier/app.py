from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import web_ft_bert_class_1k_tpl_50k_predict50k
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_smile():
    data = request.json
    smile = data.get('smile', '')
    if not smile:
        return jsonify({"error": "No SMILES string provided"}), 400
    result = web_ft_bert_class_1k_tpl_50k_predict50k.process_smiles(smile)
    return jsonify({
        "predicted_class_id": result[0],
        "rxn_str_id": result[1],
        "class_name": result[2]
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and file.filename.endswith('.tsv'):
        path = "/home/hpk/rxnfp-master/out/data/save" + file.filename
        file.save(path)
        df = pd.read_csv(path, sep='\t')
        results_df = web_ft_bert_class_1k_tpl_50k_predict50k.process_smiles_batch(df)
        output_path = "/home/hpk/rxnfp-master/out/data/output/results.csv"
        results_df.to_csv(output_path, index=False)
        return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)