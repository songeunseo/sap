"""Atomic, content-checked scientific artifacts and resumable readouts."""
import math
from pathlib import Path
from experiments.dlm_multiscale_ac50.artifacts import digest,read,write,freeze


def seal(payload):
    if 'payload_sha256' in payload:raise ValueError('Payload already sealed')
    return dict(payload,payload_sha256=digest(payload))


def verify_payload(value,expected=None):
    payload={k:v for k,v in value.items() if k!='payload_sha256'}
    if value.get('payload_sha256')!=digest(payload):raise RuntimeError('Artifact payload SHA256 mismatch')
    for k,v in (expected or {}).items():
        if payload.get(k)!=v:raise RuntimeError(f'Artifact identity mismatch: {k}')
    return payload


def read_checked(path,expected=None):return verify_payload(read(path),expected)


def freeze_checked(path,payload):
    if Path(path).exists():read_checked(path)
    freeze(path,seal(payload))
    return payload


def write_checked(path,payload):write(path,seal(payload))


class ReadoutStore:
    def __init__(self,path,fingerprint,count,width):
        self.path=Path(path);self.fingerprint=fingerprint;self.count=count;self.width=width;self.values=[]
        if self.path.exists():
            r=read_checked(path,dict(fingerprint=fingerprint,count=count,width=width));self.values=r['values']
            if r['complete']!=(len(self.values)==count):raise RuntimeError('Readout completeness mismatch')
        self._validate()
    def _validate(self):
        if len(self.values)>self.count:raise RuntimeError('Too many readouts')
        for row in self.values:
            if len(row)!=self.width or any(not math.isfinite(x) for x in row):raise RuntimeError('Nonfinite/wrong-width readout')
    def append(self,values):
        if len(self.values)>=self.count:raise RuntimeError('Readout complete')
        self.values.append([float(x) for x in values]);self._validate()
        write_checked(self.path,dict(fingerprint=self.fingerprint,count=self.count,width=self.width,values=self.values,complete=len(self.values)==self.count))
    def complete(self):
        if len(self.values)!=self.count:raise RuntimeError('Readout incomplete')
        return self.values
