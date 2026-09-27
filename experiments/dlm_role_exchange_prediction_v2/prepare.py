import json
import re
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from experiments.dlm_role_exchange_prediction_v2.common import ROOT, OLD, OLD_STORE, atomic_json, sha256

def main():
    states=json.loads((OLD/'state_manifest.json').read_text())
    design=json.loads((OLD/'design.json').read_text())
    # V1 counted every subsection heading as a new article. Reconstruct true parent articles.
    model=states['model']
    tok=AutoTokenizer.from_pretrained(model['id'],revision=model['revision'],trust_remote_code=True)
    texts=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='train')['text']
    joined=' '.join(texts)
    encoded=tok(joined,return_offsets_mapping=True)
    ids=encoded['input_ids']; offsets=encoded['offset_mapping']
    headings=[]; cursor=0
    for line in texts:
        if re.fullmatch(r'\s*=\s*[^=]+?\s*=\s*',line): headings.append(cursor)
        cursor+=len(line)+1
    import bisect
    parents={}; spans=[]
    for d in states['documents']:
        start,end=d['start'],d['end_exclusive']
        parent=bisect.bisect_right(headings,offsets[start][0])-1
        last=bisect.bisect_right(headings,offsets[end-1][1]-1)-1
        if parent<0 or parent!=last: raise RuntimeError('Span crosses article boundary')
        ref=next(s for s in states['states'] if s['sequence_index']==d['sequence_index'])
        if ids[start:end]!=ref['clean_ids'][0]: raise RuntimeError('Corpus token mismatch')
        parents[str(d['sequence_index'])]=parent; spans.append((start,end))
    dev={parents[str(i)] for i in design['development_documents']}
    final={parents[str(i)] for i in design['final_documents']}
    overlap=sorted(dev&final)
    audit={'parent_article_by_sequence':parents,'unique_parent_articles':len(set(parents.values())),
           'development_final_parent_overlap':overlap,
           'interpretation':'Correction on previously seen states; not new confirmation.'}
    ROOT.mkdir(parents=True,exist_ok=True)
    atomic_json(ROOT/'document_audit.json',audit)
    corrected_dev=[i for i in design['development_documents'] if parents[str(i)] not in final]
    if not corrected_dev:raise RuntimeError('No disjoint development documents remain')
    sources=[OLD/'design.json',OLD/'state_manifest.json',OLD/'collect.py',OLD/'analyze.py',
             OLD/'core.py',OLD_STORE/'development.pt',OLD_STORE/'final.pt',ROOT/'document_audit.json']
    for p in design['source_hashes']: sources.append(Path(p))
    cfg={'version':2,'status':'frozen_before_corrected_outcomes','study_type':'corrective exploratory rerun',
         'measurement':'batch1 full forward for all singles and bundles; physical weights; restore each intervention',
         'features':'reuse immutable v1 local full-forward reconstruction; independently check in smoke',
         'development_documents':corrected_dev,'final_documents':design['final_documents'],
         'excluded_development_documents':[i for i in design['development_documents'] if i not in corrected_dev],
         'split_correction':'retain final; exclude development spans sharing true parent article with final',
         'ridge':'equal-document-layer weighted MSE + norm2, alpha=1, intercept unpenalized',
         'bootstrap':'documents x 4-layer blocks, 20000, conditional on fitted models',
         'primary_comparisons':['P2-P1','P3-P2','P2-random'],
         'exploratory':['Pooled versus pooled+role contrast','role contrast x timestep/type'],
         'minimum_practical_relative_mse_reduction':.1,
         'interpretation_rule':'no significant gain is inconclusive; never reject role separation in general',
         'source_hashes':{str(p):sha256(p) for p in sources},
         'code_hashes':{str(p):sha256(p) for p in sorted(ROOT.glob('*.py'))}}
    out=ROOT/'config.json'
    if out.exists() and json.loads(out.read_text())!=cfg: raise RuntimeError('v2 frozen config already differs')
    atomic_json(out,cfg)
    print(json.dumps({'status':'prepared','documents':audit},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
