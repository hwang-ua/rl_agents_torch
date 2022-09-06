### IQL Offline Discrete Control
python run_linear.py --config-file experiment/config/test/four_room/iql_offline/test_linear.json --device 0 --size 10000 --is_ac 1 --id 0 --policy opt --weight-init 20.0

### CQL Offline Discrete Control
python run_linear.py --config-file experiment/config/test/four_room/cql_offline/test.json --device 0 --size 10000 --is_ac 0 --id 0 --policy opt --weight-init 20.0

### IQL Offline Continuous Control
python run_ac_offline.py --id 0 --config-file experiment/config/test_v9/halfcheetah/iql_offline/test.json