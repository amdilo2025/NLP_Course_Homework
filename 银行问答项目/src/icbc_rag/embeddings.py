"""Pinned BGE ONNX int8, CLS pooling and L2 normalization, entirely local."""
import json
import numpy as np
from icbc_baseline.bm25 import canonical_hash
from .common import file_hash,project_path

class LocalBGE:
    def __init__(self,config):
        import onnxruntime as ort
        import tokenizers
        ort.disable_telemetry_events()
        self.config=config
        folder=project_path(config['model_dir'])
        try:manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        except FileNotFoundError as exc:raise ValueError('本地向量模型未准备；请先运行模型下载工具') from exc
        if manifest['revision']!=config['model_revision']:raise ValueError('模型版本与配置不符')
        if config['model_file'] not in manifest['files'] or config['tokenizer_file'] not in manifest['files']:
            raise ValueError('模型或分词文件不在已校验清单内')
        for name,info in manifest['files'].items():
            path=(folder/name).resolve()
            if not path.is_relative_to(folder) or file_hash(path)!=info['sha256']:
                raise ValueError('本地模型文件校验失败：'+name)
        self.tokenizer=tokenizers.Tokenizer.from_file(str(folder/config['tokenizer_file']))
        self.tokenizer.no_truncation()
        pad=self.tokenizer.token_to_id('[PAD]')
        if pad is None:raise ValueError('分词器缺少PAD标记')
        self.tokenizer.enable_padding(pad_id=pad,pad_token='[PAD]')
        options=ort.SessionOptions()
        options.intra_op_num_threads=config['cpu_threads'];options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(folder/config['model_file']),sess_options=options,providers=['CPUExecutionProvider'])
        self.input_names={i.name for i in self.session.get_inputs()}
        if not {'input_ids','attention_mask'}<=self.input_names or not self.input_names<={'input_ids','attention_mask','token_type_ids'}:
            raise ValueError('ONNX输入结构不符合BERT编码器')
        outputs=[o.name for o in self.session.get_outputs()]
        if 'last_hidden_state' not in outputs:raise ValueError('ONNX缺少last_hidden_state输出')
        self.info={'model':manifest['base_model'],'conversion_repo':manifest['repo'],'revision':manifest['revision'],
          'model_sha256':manifest['files'][config['model_file']]['sha256'],
          'tokenizer_sha256':manifest['files'][config['tokenizer_file']]['sha256'],
          'pooling':'CLS','normalization':'L2','quantization':'int8','device':'CPU',
          'query_instruction':config['query_instruction'],'max_tokens':config['max_tokens'],
          'embedding_dim':config['embedding_dim'],'onnxruntime':ort.__version__,'tokenizers':tokenizers.__version__,
          'numpy':np.__version__}
        self.fingerprint=canonical_hash(self.info)

    def encode(self,texts,query=False):
        if not texts or any(not isinstance(x,str) or not x.strip() for x in texts):raise ValueError('编码输入不能为空')
        vectors=[];counts=[]
        for start in range(0,len(texts),self.config['batch_size']):
            batch=texts[start:start+self.config['batch_size']]
            if query:batch=[self.config['query_instruction']+x for x in batch]
            encoded=self.tokenizer.encode_batch(batch)
            lengths=[sum(x.attention_mask) for x in encoded]
            if max(lengths)>self.config['max_tokens']:
                raise ValueError('输入超过模型token上限；停止编码，避免静默截断业务条件或脚注')
            feeds={'input_ids':np.asarray([x.ids for x in encoded],dtype=np.int64),
              'attention_mask':np.asarray([x.attention_mask for x in encoded],dtype=np.int64),
              'token_type_ids':np.asarray([x.type_ids for x in encoded],dtype=np.int64)}
            output=self.session.run(['last_hidden_state'],{k:v for k,v in feeds.items() if k in self.input_names})[0]
            if output.ndim!=3 or output.shape[0]!=len(batch) or output.shape[2]!=self.config['embedding_dim']:
                raise ValueError('模型输出维度不符')
            pooled=output[:,0,:].astype(np.float32)
            norms=np.linalg.norm(pooled,axis=1,keepdims=True)
            if not np.isfinite(pooled).all() or (norms<=0).any():raise ValueError('模型输出含无效或零向量')
            vectors.extend((pooled/norms).tolist());counts.extend(lengths)
        return np.asarray(vectors,dtype=np.float32),counts
