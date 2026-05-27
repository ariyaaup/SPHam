import json

notebook_path = r"c:\Users\Ariya\Documents\SHam Project SSI\datates.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get('cells', [])):
    cell_type = cell.get('cell_type')
    if cell_type == 'code':
        source = cell.get('source', [])
        if isinstance(source, list):
            source = "".join(source)
        outputs = cell.get('outputs', [])
        
        # We only care about cells 14, 16, 20
        if "kmeans_clusters =" in source or "dbsc =" in source or "kmeans_ari =" in source:
            print(f"=== Cell {i+1} ({cell_type}) ===")
            print(source)
            print("--- Outputs ---")
            for out in outputs:
                if out.get('output_type') == 'stream':
                    print(out.get('text', ''))
                elif out.get('output_type') == 'display_data' or out.get('output_type') == 'execute_result':
                    data = out.get('data', {})
                    if 'text/plain' in data:
                        print(data['text/plain'])
            print("\n" + "="*40 + "\n")
