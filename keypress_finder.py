import os
import rosbag	
import rospy
import make_snippet_bag
import pickle

from keyboard.msg import Key 

# keys of interest
# Key.KEY_c 
# Key.KEY_v
# Key.KEY_l
# Key.KEY_i
# Key.KEY_s
# Key.KEY_SPACE

PICKLE_CACHE = os.path.join(os.getcwd(), 'pickle')


def make_bags(directory_name, event_log):
	"""
	"""

	top_dir = event_log['top_dir']

	for bag_name in event_log[directory_name]:
		bag_file_name = top_dir + directory_name + bag_name

		print('loading bag ' + bag_file_name)
		bag = rosbag.Bag(bag_file_name)
		start_time_secs = get_start_secs(bag)

		event_times = event_log[directory_name][bag_name]

		construction_count = 1
		for event_time in event_times:
			prefix, extension = os.path.splitext(bag_name)
			construction_bag_nme = prefix + '_' + str(construction_count) + extension
			construction_bag = rosbag.Bag(construction_bag_name, 'w+') 

			construction_count += 1

			t0 = event_time - PRECEEDING_SECONDS
			t1 = event_time + FOLLOWING_SECONDS

			for topic, msg, t in tqdm(bag.read_messages()):
				time_from_beginning = t.to_sec() - start_time_secs

				if (time_from_beginning > t0) and (time_from_beginning < t1):
					construction_bag.write(topic, msg, t)

	return



def do_stuff(bag_full_path, bag):
	"""
	"""

	bag_snaps = []
	pickle_file = '{}.pickle'.format(bag_full_path.replace('/','_'))
	pickle_out = os.path.join(PICKLE_CACHE, pickle_file)
	
	if os.path.exists(pickle_out):
		pickle_in = open(pickle_out, 'rb')
		pickle_dict = pickle.load(pickle_in)
		return pickle_dict['bag_snaps']

	for topic, msg, t in bag.read_messages():
		if topic == '/dbw/toyota_dbw/car_state':
			try:
				if msg.keypressed == Key.KEY_c:
					# print('c key pressed at {}'.format(t))
					bag_snaps.append(('C', t))

		


				if msg.keypressed == Key.KEY_v:
				 	# print('v key pressed at {}'.format(t))
					bag_snaps.append(('V', t))
			except AttributeError:
				continue


	
	
	with open(pickle_out, 'w+') as f:
		print(bag_snaps)
		pickle_dict = {'bag_snaps': bag_snaps}
		pickle.dump(pickle_dict, f)


	return bag_snaps


if __name__ == '__main__':
	"""
	"""

	# change this depending on where you have sawmill mounted
	# and what directory you want to search in

	# list of all the bag file names in top_dir
	top_dir = '/media/kuser/rack1remote/datasets/kache_ai/bags/'
	all_bag_dirs = os.listdir('/media/kuser/rack1remote/datasets/kache_ai/bags/')
	bag_data = {}
	for bag_dir in all_bag_dirs:
		bag_names = os.listdir(os.path.join(top_dir,bag_dir))

		for file_name in bag_names:
			bag_full_path = os.path.join(top_dir,bag_dir, file_name.replace(':','\:'))
			pickle_file = '{}.pickle'.format(bag_full_path.replace('/','_'))
			pickle_out = os.path.join(PICKLE_CACHE, pickle_file)
	
			if os.path.exists(pickle_out):
				pickle_in = open(pickle_out, 'rb')
				pickle_dict = pickle.load(pickle_in)
				bag_data[bag_full_path] = pickle_dict['bag_snaps']
			else:
				#print('reading bag ' + bag_full_path)
				try:
					if os.path.exists(bag_full_path):
						print('processing bag ' + bag_full_path)
						bag = rosbag.Bag(bag_full_path)
						bag_data[bag_full_path] = do_stuff(bag_full_path, bag)
					else: 
						print('Cant find {}'.format(bag_full_path))
						continue
				except rosbag.bag.ROSBagException:
					print('Ros bag {} found to be empty'.format(bag_full_path))
					bag_data[bag_full_path] = []
					continue

	with open('bag_data.pickle') as pickle_out:
		pickle_dict = {'bag_data': bag_data}
		pickle.dump(pickle_dict, pickle_out)




