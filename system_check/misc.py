# Tests that didn't neatly fit in another file. Should be deprecated eventually.

#@ sensors.depth.updating = shm.depth.depth.get() != delayed(0.5, 'shm.depth.depth.get()')
#@ sensors.depth.not_crazy = abs(shm.depth.depth.get() - delayed(0.2, 'shm.depth.depth.get()')) < 0.2
#@ artemis.sensors.pressure.valid = .7 < shm.pressure.hull.get() < .89
#@ apollo.sensors.pressure.valid = .7 < shm.pressure.hull.get() < .89
#@ sensors.pressure.updating = shm.pressure.hull.get() != delayed(0.5, 'shm.pressure.hull.get()')

#@ artemis.merge.total_voltage.ok = 16.8 > shm.merge_status.total_voltage.get() > 14.0
# #@ artemis.merge.current_starboard.ok = 30.0 > shm.merge_status.current_starboard.get() > 2.0
# #@ artemis.merge.current_port.ok = 30.0 > shm.merge_status.current_port.get() > 2.0
# #@ artemis.merge.voltage_port.ok = 16.8 > shm.merge_status.voltage_port.get() > 14.0
# #@ artemis.merge.voltage_starboard.ok = 16.8 > shm.merge_status.voltage_starboard.get() > 14.0

# #@ artemis.serial.actuator.connected = shm.connected_devices.actuator.get()
#@ artemis.serial.gpio.connected = shm.connected_devices.gpio.get()
#@ artemis.serial.merge.connected = shm.connected_devices.merge.get()
#@ artemis.serial.thrusters.connected = shm.connected_devices.thrusters.get()
#@ artemis.serial.thrusters2.connected = shm.connected_devices.thrusters2.get()
#@ artemis.serial.powerDistribution.connected = shm.connected_devices.powerDistribution.get()

# #@ apollo.serial.actuator.connected = shm.connected_devices.actuator.get()
#@ apollo.serial.gpio.connected = shm.connected_devices.gpio.get()
#@ apollo.serial.merge.connected = shm.connected_devices.merge.get()
#@ apollo.serial.thrusters.connected = shm.connected_devices.thrusters.get()
#@ apollo.serial.powerDistribution.connected = shm.connected_devices.powerDistribution.get()

# Max % CPU usage = num cores * 100%

#@ artemis.sys.cpu_usage.reasonable = float(shell('mpstat | tail -n 1 | sed "s/\s\s*/ /g" | cut -d" " -f4').stdout) < 200.0
#@ apollo.sys.cpu_usage.reasonable = float(shell('mpstat | tail -n 1 | sed "s/\s\s*/ /g" | cut -d" " -f4').stdout) < 100.0

# TODO Add back in mem usage test.

# Check for cameras.

# TODO Read these from the configuration?
#@ artemis.camera.forward.present = shell('test -f /dev/shm/auv_visiond-forward').code == 0
#@ artemis.camera.downward.present = shell('test -f /dev/shm/auv_visiond-downward').code == 0
#@ apollo.camera.forward.present = shell('test -f /dev/shm/auv_visiond-forward').code == 0
#@ apollo.camera.downward.present = shell('test -f /dev/shm/auv_visiond-downward').code == 0

#@ artemis.navigation.running = shell('auv-check-navigated').code == 0
# #@ apollo.hydrophones.pinging = shell('auv-check-pings').code == 0
# #@ artemis.hydrophones.pinging = shell('auv-check-pings').code == 0

# TODO Read from vehicle config
# @ artemis.sensors.gx1.updating = shm.threedmg.clk_ticks.get() != delayed(0.5, 'shm.threedmg.clk_ticks.get()')


