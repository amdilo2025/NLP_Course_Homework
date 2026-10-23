"""Windows user-bound encrypted storage; plaintext is never written to disk."""
import base64
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

class CredentialError(RuntimeError):pass

class Blob(ctypes.Structure):
    _fields_=[('size',wintypes.DWORD),('data',ctypes.POINTER(ctypes.c_ubyte))]

def origin(url):
    parts=urlsplit(url)
    if parts.scheme!='https' or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise CredentialError('凭据只能绑定不含账号或参数的HTTPS地址')
    return f'https://{parts.hostname.lower()}'+(':'+str(parts.port) if parts.port and parts.port!=443 else '')

def transform(data,protect):
    if os.name!='nt':raise CredentialError('此凭据存储仅适用于Windows；其他系统请使用环境变量')
    buffer=ctypes.create_string_buffer(data)
    source=Blob(len(data),ctypes.cast(buffer,ctypes.POINTER(ctypes.c_ubyte)))
    result=Blob()
    crypt=ctypes.WinDLL('crypt32',use_last_error=True)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.LocalFree.argtypes=[ctypes.c_void_p];kernel.LocalFree.restype=ctypes.c_void_p
    if protect:
        fn=crypt.CryptProtectData
        fn.argtypes=[ctypes.POINTER(Blob),wintypes.LPCWSTR,ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(Blob)]
        fn.restype=wintypes.BOOL
        ok=fn(ctypes.byref(source),'ICBC RAG API credential',None,None,None,1,ctypes.byref(result))
    else:
        fn=crypt.CryptUnprotectData
        fn.argtypes=[ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(Blob)]
        fn.restype=wintypes.BOOL
        ok=fn(ctypes.byref(source),None,None,None,None,1,ctypes.byref(result))
    if not ok:raise CredentialError('Windows凭据加密或解密失败；需由保存密钥的Windows用户操作')
    try:return ctypes.string_at(result.data,result.size)
    finally:kernel.LocalFree(ctypes.cast(result.data,ctypes.c_void_p))

def save_key(path,key,base_url):
    key=key.strip()
    if len(key)<20 or len(key)>512 or any(c.isspace() for c in key):raise CredentialError('密钥格式不完整')
    payload={'version':1,'scheme':'windows_dpapi_current_user','origin':origin(base_url),
      'ciphertext':base64.b64encode(transform(key.encode('utf-8'),True)).decode('ascii')}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(payload,indent=2),encoding='utf-8');temporary.replace(path)

def load_key(path,base_url):
    path=Path(path)
    if not path.is_file():return None
    try:
        payload=json.loads(path.read_text(encoding='utf-8'))
        if payload['version']!=1 or payload['scheme']!='windows_dpapi_current_user':raise CredentialError('凭据格式不支持')
        if payload['origin']!=origin(base_url):raise CredentialError('保存的密钥不允许用于此API主机')
        return transform(base64.b64decode(payload['ciphertext'],validate=True),False).decode('utf-8')
    except (ValueError,KeyError,UnicodeError) as exc:
        raise CredentialError('本地加密凭据无法读取，请重新配置') from None
