import json

nb = json.load(open(r'C:\Users\Utente\Documents\PycharmProjects\dl-RAG-denseAndKG\answer_curation.ipynb', encoding='utf-8'))
for c in nb['cells']:
    if c.get('id') in ('287ddd56', 'e85c000d'):
        src = c['source']
        if isinstance(src, list):
            text = ''.join(src)
        else:
            text = src
        print(f'=== Cell ID: {c["id"]} ===')
        print(text)
        print('=== END ===')
        print()