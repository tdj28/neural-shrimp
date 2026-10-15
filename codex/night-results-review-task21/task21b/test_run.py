"""Synthetic contract controls only; no target paths, spectra or producer code."""
import json,hashlib,tempfile,unittest
from pathlib import Path
import run as R

class Controls(unittest.TestCase):
    def fixture(self,base):
        def write(name,obj,size=None):
            p=base/name;b=json.dumps(obj).encode()
            if size is not None:
                self.assertLessEqual(len(b),size);b+=b' '*(size-len(b))
            p.write_bytes(b);return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
        definitions=[('KY1',list(range(20)),list(range(18)),[1000,2000],{'high':list(range(10)),'head':list(range(10,20))},11971),
            ('KY2',list(range(10)),list(range(10)),[1000,2000],{'host':list(range(10))},6771),
            ('TC',[10,47,56,143,240,293,242,406],[10,47,56,143,240,293,242],[250,500,1000,2000],{'prior_headline':[10,47,56,240,293,242],'periodic_control':[143,406]},7275)]
        source=write('source',{});runs=[]
        for name,requested,completed,steps,groups,size in definitions:
            coordinates=[[i/1000,.1] for i in range(max(requested)+1)]
            args={'cells_index':requested,'k':4,'record':2000,'transient':1000,'checkpoints':steps,'model':'synthetic'}
            vals=[.1,-.2,-.3,-.4]
            cells=[dict(index=i,k=4,log10_temperature=coordinates[i][0],alpha=.1,exponents=vals,
                checkpoints=[dict(steps=s,exponents=vals) for s in steps],entropy_mean=1) for i in completed]
            f=write(name,dict(args=args,cells=cells,complete=len(requested)==len(completed)),size)
            runs.append(dict(id=name,file=f,cells_file=write(name+'-cells',{'cells':coordinates}),source=source,
                requested_indices=requested,completed_indices=completed,missing_indices=requested[len(completed):],
                requested_count=len(requested),completed_count=len(completed),checkpoint_steps=steps,
                k=4,transient=1000,record=2000,expected_complete=len(requested)==len(completed),original_args_metadata=args,
                recorded_termination_verified=True,selected_coordinates=[{'index':i,'coordinates':coordinates[i]} for i in requested],
                group_memberships=groups,original_disposition='synthetic',source_commit='synthetic',leading_spectrum_authority='synthetic leading estimates',ambient_dimension=896,ambient_dimension_scope='synthetic'))
        manifest={'authority':write('authority',{}),'source_metadata_manifest':write('metadata',{}),'runs':runs,'requested_cells':38,'completed_cells':35,'selected_file_bytes':26017}
        card={'input_manifest':write('inputs',manifest),'selection_policy':{x['id']:'synthetic' for x in runs},
              'group_membership_basis':{k:'synthetic' for k in ['high','head','host','prior_headline','periodic_control','headline_union','all_requested']}}
        return card,manifest,write
    def test_complete_shape_with_missing_cells_retained(self):
        with tempfile.TemporaryDirectory() as t:
            card,_,_=self.fixture(Path(t));r=R.calculate(card)
            self.assertEqual(r['completed_cell_count'],35)
            self.assertEqual(r['groups']['headline_union']['total_completed_rows'],14)
            self.assertEqual(r['groups']['periodic_control']['missing_requested_row_keys'],['TC:406'])
            self.assertEqual(r['groups']['all_requested']['missing_requested_row_keys'],['KY1:18','KY1:19','TC:406'])
            self.assertEqual(len([c for row in r['rows'] for c in row['checkpoints']]),84)
            R.encode(r)
    def test_original_complete_flag_refuses(self):
        with tempfile.TemporaryDirectory() as t:
            card,m,write=self.fixture(Path(t));m['runs'][2]['expected_complete']=True
            card['input_manifest']=write('inputs',m)
            with self.assertRaisesRegex(ValueError,'complete flag'):R.calculate(card)
    def test_changed_selected_pin_refuses(self):
        with tempfile.TemporaryDirectory() as t:
            card,m,_=self.fixture(Path(t));Path(m['runs'][0]['file']['path']).write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError,'input pin mismatch'):R.calculate(card)

if __name__=='__main__':unittest.main()
