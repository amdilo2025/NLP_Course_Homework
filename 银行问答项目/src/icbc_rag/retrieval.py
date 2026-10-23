import json
import numpy as np
from icbc_baseline.bm25 import canonical_hash
from .common import project_path,read_jsonl

def article_context(documents,hits):
    """Attach available sibling paragraphs, preserving each original source ID.

    Ranking stays paragraph-based; sibling paragraphs supply governing conditions
    and exceptions, and are not reported as new similarity search hits.
    """
    context=[];seen=set()
    for hit in hits:
        if hit.get('article_number') is None:related=[hit]
        else:
            related=sorted([d for d in documents if d.get('source_id')==hit.get('source_id')
                and d.get('article_number')==hit['article_number']],key=lambda d:d.get('paragraph_number',0))
        for doc in related:
            if doc['chunk_id'] not in seen:
                context.append(doc);seen.add(doc['chunk_id'])
    return context

class DenseIndex:
    def __init__(self,documents,vectors,encoder_fingerprint,counts):
        if not documents or len({d['chunk_id'] for d in documents})!=len(documents):raise ValueError('语料为空或ID重复')
        matrix=np.asarray(vectors,dtype=np.float32)
        if matrix.ndim!=2 or matrix.shape[0]!=len(documents) or not np.isfinite(matrix).all():raise ValueError('索引向量形状或数值无效')
        if not np.allclose(np.linalg.norm(matrix,axis=1),1,atol=1e-5):raise ValueError('索引向量必须已归一化')
        if len(counts)!=len(documents):raise ValueError('分词计数与片段数不符')
        self.documents=documents;self.vectors=matrix
        self.encoder_fingerprint=encoder_fingerprint;self.counts=counts
        self.corpus_sha256=canonical_hash(documents)

    def payload(self):
        data={'schema_version':1,'corpus_sha256':self.corpus_sha256,
          'encoder_fingerprint':self.encoder_fingerprint,'documents':self.documents,
          'vectors':self.vectors.tolist(),'token_counts':self.counts}
        return {**data,'payload_sha256':canonical_hash(data)}

    def save(self,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.json.part')
        temporary.write_text(json.dumps(self.payload(),ensure_ascii=False),encoding='utf-8')
        temporary.replace(path)

    @classmethod
    def load(cls,path,documents,fingerprint):
        data=json.loads(path.read_text(encoding='utf-8'))
        digest=data.pop('payload_sha256')
        if canonical_hash(data)!=digest:raise ValueError('向量索引文件完整性校验失败')
        if data['schema_version']!=1 or data['corpus_sha256']!=canonical_hash(documents) or data['documents']!=documents:
            raise ValueError('向量索引语料版本不符，请显式重建')
        if data['encoder_fingerprint']!=fingerprint:raise ValueError('编码器版本或参数变化，请显式重建向量索引')
        return cls(documents,data['vectors'],fingerprint,data['token_counts'])

    def search_vector(self,query,vector,top_k):
        if isinstance(top_k,bool) or not isinstance(top_k,int) or not 1<=top_k<=len(self.documents):raise ValueError('top_k无效')
        v=np.asarray(vector,dtype=np.float32)
        if v.shape!=(self.vectors.shape[1],) or not np.isfinite(v).all() or not np.isclose(np.linalg.norm(v),1,atol=1e-5):
            raise ValueError('查询向量维度或归一化无效')
        scores=self.vectors@v
        order=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),self.documents[i]['chunk_id']))[:top_k]
        return {'query':query,'retrieval_only':True,'score_kind':'cosine_similarity_not_confidence',
          'results':[{'rank':rank,'score':float(scores[i]),**self.documents[i]} for rank,i in enumerate(order,1)]}

def prepare_index(config,encoder,rebuild=False):
    docs=read_jsonl(project_path(config['corpus_file']))
    if canonical_hash(docs)!=config['corpus_sha256']:raise ValueError('固定语料哈希变化，停止使用')
    path=project_path(config['index_file'])
    if path.is_file() and not rebuild:
        return DenseIndex.load(path,docs,encoder.fingerprint),False
    vectors,counts=encoder.encode([d['text'] for d in docs])
    index=DenseIndex(docs,vectors,encoder.fingerprint,counts);index.save(path)
    return index,True
