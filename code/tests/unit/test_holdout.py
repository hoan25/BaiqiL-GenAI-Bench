import pytest
from scripts.evaluate_holdout import verify_files, evaluate
from src.data.acceptance import sha256


def test_freeze_rejects_changed_file_and_escape(tmp_path):
    file=tmp_path/'head.pt'
    file.write_bytes(b'original')
    protocol={'selected':'mlp','seed':42,'min_label_gap':0.5,
              'files_sha256':{'head.pt':sha256(file)}}
    verify_files(tmp_path,protocol)
    file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Frozen input changed'):
        verify_files(tmp_path,protocol)
    protocol['files_sha256']={'../escape.pt':'anything'}
    with pytest.raises(ValueError,match='Frozen input changed'):
        verify_files(tmp_path,protocol)


def test_existing_result_is_not_overwritten(tmp_path):
    output=tmp_path/'holdout.json'
    output.write_text('existing')
    with pytest.raises(FileExistsError):
        evaluate(tmp_path,tmp_path/'missing_protocol.json',output)
    assert output.read_text()=='existing'
