"""Store a key with Windows DPAPI. No network request is sent by this tool."""
import argparse
import getpass
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from icbc_rag.credentials import save_key,load_key

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stdin-key',action='store_true')
    args=parser.parse_args()
    key=sys.stdin.readline().strip() if args.stdin_key else getpass.getpass('DeepSeek API key（不回显）: ')
    path=ROOT/'.secrets/deepseek.dpapi.json'
    save_key(path,key,'https://api.deepseek.com')
    if load_key(path,'https://api.deepseek.com')!=key:raise RuntimeError('本机凭据回读检查失败')
    print('DeepSeek密钥已加密保存并通过本机回读验证；没有写入源码或日志，没有发送网络请求。')

if __name__=='__main__':main()
