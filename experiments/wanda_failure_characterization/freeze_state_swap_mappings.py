import hashlib,json
from pathlib import Path

import torch

from experiments.wanda_failure_characterization.propagation_core import cyclic_donor_map,same_timestep_donor_map

ROOT=Path(__file__).parent

def main():
    held=json.loads((ROOT/'heldout_state_manifest.json').read_text());old=torch.load(ROOT/'failure_map.pt',map_location='cpu',weights_only=True)['damage'];count=len(held['states'])
    sequence=old['sequence_index'].tolist();timestep=old['timestep_index'].tolist();cyclic=cyclic_donor_map(count);same=same_timestep_donor_map(sequence,timestep)
    entries=[]
    for i,state in enumerate(held['states']):
        mask_count=sum(state['mask'][0]);base={'receiver':i,'receiver_sequence':sequence[i],'receiver_timestep_index':timestep[i],'receiver_p_mask':state['p_mask'],'receiver_mask_count':mask_count}
        for kind,donor in (('cyclic',cyclic[i]),('same_timestep',same[i])):
            dstate=held['states'][donor];entries.append({**base,'mapping':kind,'donor':donor,'donor_sequence':sequence[donor],'donor_timestep_index':timestep[donor],'donor_p_mask':dstate['p_mask'],'donor_mask_count':sum(dstate['mask'][0])})
    core={'state_count':count,'heldout_sha':held['historical_state_sha256'],'rules':{'cyclic':'donor(i)=(i+1) mod 40','same_timestep':'next sequence in sorted cyclic sequence order at identical timestep'},'cyclic_donors':cyclic,'same_timestep_donors':same,'entries':entries,'frozen_before_swap_evaluation':True}
    encoded=json.dumps(core,sort_keys=True,separators=(',',':')).encode();core['mapping_sha256']=hashlib.sha256(encoded).hexdigest();path=ROOT/'state_swap_mapping.json';tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(core,indent=2,sort_keys=True)+'\n');tmp.replace(path);print(core['mapping_sha256'])

if __name__=='__main__':main()
