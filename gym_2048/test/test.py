import unittest
import sys, os
sys.path.append(os.pardir)
import gym
import gym_2048
import numpy as np

env = gym.make('2048-v0')
env.reset()

class Test2048(unittest.TestCase):
    def test_set_table(self):
        board = np.array([
            [0,4,2,2],
            [16,0,2048,2],
            [0,0,2048,4],
            [4,4,8,4]
        ])
        env.set_table(board)
        self.assertEqual(np.array_equal(env.table, board), True)

    def test_up(self):
        board = np.array([
            [0,4,2,2],
            [16,0,2048,2],
            [0,0,2048,4],
            [4,4,8,4]
        ])
        env.set_table(board)
        actual_board, actual_reward = env.up()
        expected_board = np.array([
            [16,8,2,4],
            [4,0,4096,8],
            [0,0,8,0],
            [0,0,0,0]
        ])
        expected_reward = 4116
        self.assertEqual(np.array_equal(expected_board,actual_board), True)
        self.assertEqual(expected_reward, actual_reward)

    def test_right(self):
        board = np.array([
            [0,2,0,0],
            [16,0,4,0],
            [0,0,4,2],
            [4,4,8,8]
        ])
        env.set_table(board)
        actual_board, actual_reward = env.right()
        expected_board = np.array([
            [0,0,0,2],
            [0,0,16,4],
            [0,0,4,2],
            [0,0,8,16]
        ])
        expected_reward = 24
        self.assertEqual(np.array_equal(expected_board,actual_board), True)
        self.assertEqual(expected_reward, actual_reward)
    
    def test_down(self):
        board = np.array([
            [2,2,512,16],
            [16,0,512,0],
            [0,0,32768,2],
            [4,2,32768,8]
        ])
        env.set_table(board)
        actual_board, actual_reward = env.down()
        expected_board = np.array([
            [0,0,0,0],
            [2,0,0,16],
            [16,0,1024,2],
            [4,4,65536,8]
        ])
        expected_reward = 66564
        self.assertEqual(np.array_equal(expected_board,actual_board), True)
        self.assertEqual(expected_reward, actual_reward)
    
    def test_left(self):
        board = np.array([
            [2,1024,1024,4],
            [16,0,512,0],
            [0,2,2,4],
            [0,0,32,8]
        ])
        env.set_table(board)
        actual_board, actual_reward = env.left()
        expected_board = np.array([
            [2,2048,4,0],
            [16,512,0,0],
            [4,4,0,0],
            [32,8,0,0]
        ])
        expected_reward = 2052
        self.assertEqual(np.array_equal(expected_board,actual_board), True)
        self.assertEqual(expected_reward, actual_reward)

if __name__ == "__main__":
    unittest.main()