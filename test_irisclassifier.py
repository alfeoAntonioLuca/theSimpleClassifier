from typing import Set
import irisclassifier
import pytest

epochs: Set[int] = {5, 10, 25, 50, 100, 200}


@pytest.mark.parametrize('epoch', epochs)
def test_evaluation(epoch):
    i = irisclassifier.IrisClassifier(epoch)
    assert i.evaluation() > 0.75
