"""A failed placement returns the instance to the queue holding nothing.

A re-evaluated SCHEDULED instance keeps its worker assignment and subordinate
claims unless the no-candidate write-back clears them. Subordinate claims are
counted by `compute_worker_allocated` whatever the instance state, and the
pending retry predicate requires `worker_id is None`.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpustack.policies.resource_view import compute_worker_allocated
from gpustack.scheduler.scheduler import Scheduler
from gpustack.schemas.models import (
    ComputedResourceClaim,
    DistributedServers,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
)
from tests.utils.model import new_model, new_model_instance


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Service:
    def __init__(self, session):
        pass

    async def update(self, instance):
        return instance


@pytest.mark.asyncio
@pytest.mark.parametrize("assigned", [False, True], ids=["unassigned", "distributed"])
async def test_failed_reschedule_releases_queue_claims(monkeypatch, assigned):
    old = datetime.now(timezone.utc) - timedelta(minutes=10)
    inst = new_model_instance(
        id=7,
        name="dist",
        model_id=1,
        worker_id=11 if assigned else None,
        state=ModelInstanceStateEnum.ANALYZING,
        gpu_indexes=[0],
        computed_resource_claim=ComputedResourceClaim(ram=1, vram={0: 7_000_000_000}),
    )
    inst.gpu_type = "cuda"
    inst.worker_name = "main"
    inst.created_at = old - timedelta(hours=1)
    inst.updated_at = old
    if assigned:
        inst.distributed_servers = DistributedServers(
            subordinate_workers=[
                ModelInstanceSubordinateWorker(
                    worker_id=22,
                    worker_name="sub",
                    gpu_indexes=[1],
                    gpu_type="cuda",
                    computed_resource_claim=ComputedResourceClaim(
                        ram=0, vram={1: 7_000_000_000}
                    ),
                )
            ]
        )
    model = new_model(1, "m", huggingface_repo_id="org/m", replicas=1)

    async def workers(session):
        return [SimpleNamespace(id=11)]

    async def one_model(session, _id):
        return model

    async def one_inst(session, _id):
        return inst

    async def all_inst(session, options=None):
        return [inst]

    async def no_candidate(*a, **k):
        return None, ["No workers meet the resource requirements.\n"]

    p = "gpustack.scheduler.scheduler."
    monkeypatch.setattr(p + "async_session", lambda: _Session())
    monkeypatch.setattr(p + "Worker.all", workers)
    monkeypatch.setattr(p + "Model.one_by_id", one_model)
    monkeypatch.setattr(p + "ModelInstance.one_by_id", one_inst)
    monkeypatch.setattr(p + "ModelInstance.all", all_inst)
    monkeypatch.setattr(p + "find_candidate", no_candidate)
    monkeypatch.setattr(p + "ModelInstanceService", _Service)

    scheduler = Scheduler(SimpleNamespace(cache_dir=None))
    await scheduler._schedule_one(inst)

    assert inst.state == ModelInstanceStateEnum.PENDING
    assert inst.computed_resource_claim is None
    assert inst.worker_id is None
    assert inst.worker_name is None
    assert not inst.gpu_indexes
    assert inst.distributed_servers is None
    assert compute_worker_allocated([inst], 22, "cuda").vram == {}

    inst.updated_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    assert scheduler._should_schedule(inst) is True
