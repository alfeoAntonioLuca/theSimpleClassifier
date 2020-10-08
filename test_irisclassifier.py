from typing import Set
import irisclassifier
import pytest

epochs: Set[int] = {50, 100, 200, 300, 500}


@pytest.mark.parametrize('epoch', epochs)
def test_evaluations(epoch):
    i = irisclassifier.IrisClassifier(epoch)
    i.ingestion()
    i.segregation()
    i.train()
    res = i.evaluation()
    assert res > 0.75


def test_evaluation():
    i = irisclassifier.IrisClassifier()
    i.ingestion()
    i.segregation()
    i.train()
    res = i.evaluation()
    assert res > 0.75
