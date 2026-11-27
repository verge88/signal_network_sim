import unittest
from ss7.hierarchical_kie_v8_sharded import PLAN,SHARDS,shard_plan
class ShardTests(unittest.TestCase):
    def test_partition(self):
        actual=[p for i in range(SHARDS) for p in shard_plan(i)]
        self.assertEqual(sorted(actual),sorted(PLAN))
        self.assertEqual(len(set(actual)),len(PLAN))
        self.assertTrue(all(len(shard_plan(i))==3 for i in range(SHARDS)))
    def test_invalid_shard(self):
        with self.assertRaises(ValueError): shard_plan(SHARDS)
if __name__=="__main__": unittest.main()
