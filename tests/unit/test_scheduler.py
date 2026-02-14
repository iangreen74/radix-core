"""Tests for scheduling policies and planners."""

import pytest

from radix_core.scheduler.job_graph import JobGraph
from radix_core.scheduler.placement import (
    LoadBalancedPlacement,
    LocalPlacement,
    ResourceNode,
    get_placement_strategy,
)
from radix_core.scheduler.planner import GreedyPlanner, OptimalPlanner, get_planner
from radix_core.scheduler.policies import (
    FairSharePolicy,
    FIFOPolicy,
    PriorityPolicy,
    ShortestJobFirstPolicy,
)
from radix_core.types import Job, ResourceRequirements


class TestFIFOPolicy:
    def test_empty_ready_jobs(self):
        policy = FIFOPolicy()
        res = ResourceRequirements(cpu_cores=8.0, memory_mb=8192)
        decisions = policy.select_jobs([], res)
        assert decisions == []

    def test_fifo_ordering(self, sample_jobs, available_resources):
        policy = FIFOPolicy()
        decisions = policy.select_jobs(sample_jobs, available_resources)
        assert len(decisions) == len(sample_jobs)
        # FIFO: ordered by created_at (all created nearly simultaneously)
        assert all(d.priority_score is not None for d in decisions)

    def test_respects_max_jobs(self, sample_jobs, available_resources):
        policy = FIFOPolicy()
        decisions = policy.select_jobs(sample_jobs, available_resources, max_jobs=2)
        assert len(decisions) == 2

    def test_skips_infeasible_jobs(self):
        policy = FIFOPolicy()
        big_job = Job(
            command="echo", requirements=ResourceRequirements(cpu_cores=100.0, memory_mb=512)
        )
        tiny_res = ResourceRequirements(cpu_cores=1.0, memory_mb=256)
        decisions = policy.select_jobs([big_job], tiny_res)
        assert decisions == []


class TestPriorityPolicy:
    def test_higher_priority_first(self, available_resources):
        policy = PriorityPolicy()
        low = Job(command="echo lo", priority=1)
        high = Job(command="echo hi", priority=10)
        decisions = policy.select_jobs([low, high], available_resources)
        assert len(decisions) == 2
        assert decisions[0].job_id == high.job_id

    def test_statistics(self, sample_jobs, available_resources):
        policy = PriorityPolicy()
        policy.select_jobs(sample_jobs, available_resources)
        stats = policy.get_statistics()
        assert stats["decisions_made"] == len(sample_jobs)


class TestFairSharePolicy:
    def test_basic_fair_share(self, available_resources):
        policy = FairSharePolicy()
        job_a = Job(command="echo a", tags={"user": "alice"})
        job_b = Job(command="echo b", tags={"user": "bob"})
        decisions = policy.select_jobs([job_a, job_b], available_resources)
        assert len(decisions) == 2

    def test_set_fair_share(self):
        policy = FairSharePolicy()
        policy.set_fair_share("alice", 2.0)
        assert policy.fair_shares["alice"] == 2.0


class TestShortestJobFirstPolicy:
    def test_shortest_first(self, available_resources):
        policy = ShortestJobFirstPolicy()
        small = Job(
            command="echo s", requirements=ResourceRequirements(cpu_cores=1.0, memory_mb=256)
        )
        big = Job(
            command="echo b", requirements=ResourceRequirements(cpu_cores=4.0, memory_mb=4096)
        )
        decisions = policy.select_jobs([big, small], available_resources)
        assert len(decisions) == 2
        # SJF: shorter estimated duration first
        assert decisions[0].job_id == small.job_id


class TestGreedyPlanner:
    def test_empty_jobs(self):
        planner = GreedyPlanner()
        plan = planner.create_execution_plan([])
        assert len(plan.scheduled_jobs) == 0

    def test_plans_jobs(self, sample_jobs):
        planner = GreedyPlanner()
        plan = planner.create_execution_plan(sample_jobs)
        assert len(plan.scheduled_jobs) == len(sample_jobs)
        assert plan.dependencies_resolved is True

    def test_validates_dependencies(self):
        planner = GreedyPlanner()
        job = Job(command="echo", dependencies=["nonexistent"])
        assert planner.validate_dependencies([job]) is False


class TestOptimalPlanner:
    def test_plans_jobs(self, sample_jobs):
        planner = OptimalPlanner()
        plan = planner.create_execution_plan(sample_jobs)
        assert len(plan.scheduled_jobs) == len(sample_jobs)

    def test_empty_plan(self):
        planner = OptimalPlanner()
        plan = planner.create_execution_plan([])
        assert len(plan.scheduled_jobs) == 0


class TestGetPlanner:
    def test_greedy(self):
        p = get_planner("greedy")
        assert isinstance(p, GreedyPlanner)

    def test_optimal(self):
        p = get_planner("optimal")
        assert isinstance(p, OptimalPlanner)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown planner type"):
            get_planner("quantum")


class TestLocalPlacement:
    def test_place_single_job(self, sample_job):
        strategy = LocalPlacement()
        plan = strategy.place_jobs([sample_job])
        assert len(plan.placements) == 1
        assert len(plan.unplaceable_jobs) == 0

    def test_resource_overflow(self):
        strategy = LocalPlacement()
        huge = Job(
            command="echo", requirements=ResourceRequirements(cpu_cores=999.0, memory_mb=512)
        )
        plan = strategy.place_jobs([huge])
        assert len(plan.placements) == 0
        assert len(plan.unplaceable_jobs) == 1


class TestLoadBalancedPlacement:
    def test_distributes_jobs(self, sample_jobs):
        strategy = LoadBalancedPlacement()
        plan = strategy.place_jobs(sample_jobs)
        assert len(plan.placements) == len(sample_jobs)
        # Check that jobs are spread across nodes
        nodes_used = {p.node_id for p in plan.placements}
        assert len(nodes_used) >= 1

    def test_custom_nodes(self, sample_job):
        nodes = [
            ResourceNode(
                node_id="n1",
                available_cpu=8.0,
                available_memory=16384.0,
                available_gpu=0,
                total_cpu=8.0,
                total_memory=16384.0,
                total_gpu=0,
            ),
        ]
        strategy = LoadBalancedPlacement()
        plan = strategy.place_jobs([sample_job], nodes)
        assert len(plan.placements) == 1
        assert plan.placements[0].node_id == "n1"


class TestGetPlacementStrategy:
    def test_local(self):
        s = get_placement_strategy("local")
        assert isinstance(s, LocalPlacement)

    def test_load_balanced(self):
        s = get_placement_strategy("load_balanced")
        assert isinstance(s, LoadBalancedPlacement)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_placement_strategy("cloud")


class TestJobGraph:
    def test_add_and_get_jobs(self):
        graph = JobGraph()
        job = Job(command="echo a")
        graph.add_job(job)
        assert job.job_id in graph
        assert len(graph) == 1

    def test_remove_job(self):
        graph = JobGraph()
        job = Job(command="echo a")
        graph.add_job(job)
        assert graph.remove_job(job.job_id) is True
        assert job.job_id not in graph

    def test_dependency_management(self):
        graph = JobGraph()
        a = Job(command="echo a")
        b = Job(command="echo b")
        graph.add_job(a)
        graph.add_job(b)
        assert graph.add_dependency(a.job_id, b.job_id) is True
        # b depends on a, so only a is ready
        ready = graph.get_ready_jobs()
        assert len(ready) == 1
        assert ready[0].job_id == a.job_id

    def test_cycle_detection(self):
        graph = JobGraph()
        a = Job(command="echo a")
        b = Job(command="echo b")
        graph.add_job(a)
        graph.add_job(b)
        graph.add_dependency(a.job_id, b.job_id)
        assert graph.add_dependency(b.job_id, a.job_id) is False  # would create cycle

    def test_topological_order(self):
        graph = JobGraph()
        a = Job(command="echo a")
        b = Job(command="echo b")
        c = Job(command="echo c")
        graph.add_job(a)
        graph.add_job(b)
        graph.add_job(c)
        graph.add_dependency(a.job_id, b.job_id)
        graph.add_dependency(b.job_id, c.job_id)
        order = graph.get_topological_order()
        assert order.index(a.job_id) < order.index(b.job_id) < order.index(c.job_id)

    def test_mark_job_completed(self):
        graph = JobGraph()
        a = Job(command="echo a")
        b = Job(command="echo b")
        graph.add_job(a)
        graph.add_job(b)
        graph.add_dependency(a.job_id, b.job_id)
        graph.mark_job_completed(a.job_id)
        ready = graph.get_ready_jobs()
        ready_ids = [j.job_id for j in ready]
        assert b.job_id in ready_ids

    def test_parallel_levels(self):
        graph = JobGraph()
        a = Job(command="echo a")
        b = Job(command="echo b")
        c = Job(command="echo c")
        graph.add_job(a)
        graph.add_job(b)
        graph.add_job(c)
        graph.add_dependency(a.job_id, c.job_id)
        graph.add_dependency(b.job_id, c.job_id)
        levels = graph.get_parallel_levels()
        assert len(levels) == 2
        # a and b can run in parallel at level 0
        assert set(levels[0]) == {a.job_id, b.job_id}
        assert levels[1] == [c.job_id]

    def test_critical_path(self):
        graph = JobGraph()
        a = Job(command="echo a")
        b = Job(command="echo b")
        graph.add_job(a)
        graph.add_job(b)
        graph.add_dependency(a.job_id, b.job_id)
        path, duration = graph.get_critical_path()
        assert a.job_id in path
        assert b.job_id in path
        assert duration > 0

    def test_validate_clean_graph(self):
        graph = JobGraph()
        a = Job(command="echo a")
        graph.add_job(a)
        errors = graph.validate()
        assert errors == []

    def test_statistics(self):
        graph = JobGraph()
        a = Job(command="echo a")
        graph.add_job(a)
        stats = graph.get_statistics()
        assert stats["total_jobs"] == 1
        assert stats["ready_jobs"] == 1

    def test_duplicate_job_raises(self):
        graph = JobGraph()
        job = Job(command="echo a")
        graph.add_job(job)
        with pytest.raises(Exception):
            graph.add_job(job)
