import numpy as np
import pytest
from src.models.interaction_features import interaction_features


def test_alignment_changes_features_and_preserves_batch_order():
    text=np.eye(2,dtype=np.float32)
    image=text[::-1].copy()
    assert interaction_features(text,text,'cosine').ravel().tolist()==[1,1]
    assert interaction_features(text,image,'cosine').ravel().tolist()==[0,0]
    actual=interaction_features(text,image,'interaction')
    assert actual.shape==(2,5)
    np.testing.assert_array_equal(actual, [[0,0,0,1,1],[0,0,0,1,1]])
    np.testing.assert_array_equal(actual[:1],interaction_features(text[:1],image[:1],'interaction'))
    hybrid=interaction_features(text,image,'hybrid')
    np.testing.assert_array_equal(hybrid[:,:2],text)
    np.testing.assert_array_equal(hybrid[:,2:4],image)
    np.testing.assert_array_equal(hybrid[:,4:],actual)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        interaction_features([[float('nan')]],[[1]],'cosine')
    with pytest.raises(ValueError):
        interaction_features([[1]],[[1,2]],'cosine')
    with pytest.raises(ValueError):
        interaction_features([[1]],[[1]],'unknown')
