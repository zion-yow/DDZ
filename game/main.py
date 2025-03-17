import random
from enum import Enum
from collections import Counter
from itertools import groupby
from cards import Suit, Rank, Card, Player
import engine as engine
import PCplayer as PCplayer
import state_encoder as encoder  # 导入状态编码模块
import time  # 导入time模块用于AI对战时的延迟显示


class DouDiZhuGame:
    def __init__(self, game_mode='human_vs_ai', agent=None):
        self.agent = agent
        self.game_mode = game_mode  # 游戏模式：'human_vs_ai' 或 'ai_vs_ai'
        
        # 根据游戏模式设置玩家
        if game_mode == 'human_vs_ai':
            self.players = [Player("You",''), Player("PCPlayer1", 'peasant1'), Player("PCPlayer2", 'peasant2')]
        else:  # AI对战模式
            self.players = [Player("AI-1",''), Player("AI-2", 'peasant1'), Player("AI-3", 'peasant2')]
        
        self.deck = [] # 牌堆
        self.current_player = 0
        self.last_played = [] # 上家出的牌
        self.landlord_cards = []
        self.continued_passed_count = 0
        self.create_deck()
        self.play_history = []  # 记录出牌历史，格式为[(player_index, cards), ...]
        self.game_state = {}  # 存储游戏状态
        

    def create_deck(self):
        """生成牌库"""
        for suit in [Suit.SPADE, Suit.HEART, Suit.CLUB, Suit.DIAMOND]:
            for rank in Rank:
                if rank.value <= 15:  # 排除大小王
                    self.deck.append(Card(suit, rank))
        # 添加大小王
        self.deck.append(Card(Suit.JOKER, Rank.SMALL_JOKER))
        self.deck.append(Card(Suit.JOKER, Rank.BIG_JOKER))

    
    def shuffle_and_deal(self):
        """洗牌并发牌"""
        random.shuffle(self.deck, random.seed(1))
        for i in range(3):
            start = i * 17
            end = start + 17
            self.players[i].hand = self.deck[start:end]
        self.landlord_cards = self.deck[51:]

        
    def determine_landlord(self):
        """确定地主"""
        landlord = random.randint(0,2)
        self.players[landlord].role = 'landlord'
        self.players[landlord].hand.extend(self.landlord_cards)

        self.players[landlord].sort_hand()
        for player in self.players:
            player.sort_hand()
            if player.name == 'You':
                continue
            else:
                player.name == player.role
            
        self.current_player = landlord

        print(f"{self.players[landlord].name} becomes landlord!")
        self.players[landlord].name = self.players[landlord].name + ' (LANDLORD)'

        # 更新游戏状态
        self.update_game_state()


    def playing(self, current_player, cardindexes):
        """出牌"""
        indexes = cardindexes.copy()
        played_cards = [current_player.hand[i] for i in indexes] # 出的牌
        valider = engine.validate_play(played_cards, self.last_played)

        if valider:
            # 移除手牌
            print(current_player.hand)
            # 记录出牌历史
            player_index = self.players.index(current_player)
            self.play_history.append((player_index, played_cards))

            for i in sorted(indexes, reverse=True):
                del current_player.hand[i]
            self.last_played = played_cards
            
            #  连续过牌次数清零
            self.continued_passed_count = 0
            # 展示出的牌
            print(played_cards)
            # 更新游戏状态
            self.update_game_state()

            return True, played_cards

        else:
            print(played_cards)
            print("Invalid play! Try again.")

            return False, played_cards

    def ai_player_playing(self, current_player, action=None):
        """AI的选牌，支持RL智能体动作"""
        if action is not None:  # RL智能体动作
            played = action
        else:  # 原始PC玩家逻辑
            played = PCplayer.play(self.last_played, current_player.hand, power=1)
        
        valider = engine.validate_play(played, self.last_played)
        
        if valider:
            # 移除手牌
            if played and len(played) > 0:  # 确保played不为空
                for card in played:
                    if card in current_player.hand:
                        current_player.hand.remove(card)
                self.last_played = played

                #  连续过牌次数清零
                self.continued_passed_count = 0
                # 展示出的牌
                print(played)
                
                # 记录出牌历史
                player_index = self.players.index(current_player)
                self.play_history.append((player_index, played))

                # 更新游戏状态
                self.update_game_state()
                
                return True, played
            else:
                if not self.last_played:  # 作为第一个出牌人，未出牌时不能过
                    return False, played
                else:
                    self.continued_passed_count += 1
                    print('pass')

                    # 记录过牌历史
                    player_index = self.players.index(current_player)
                    self.play_history.append((player_index, []))
                    
                    # 更新游戏状态
                    self.update_game_state()
                    
                    return True, played
        else:
            print(played)
            print("Invalid play! Try again.")
            return False, played


    def update_game_state(self):
        """更新游戏状态"""
        self.game_state = encoder.get_obs(self)


    def play_round(self, winner):
        """一轮游戏"""
        current_player = self.players[self.current_player]
        valid_play = False

        while not valid_play and not winner:
            print(current_player.name)
            print(f"\n{current_player.name}'s turn")
            print( current_player.hand) # 假设AI明牌

            current_state = self.game_state[self.current_player]

            # 人机对战模式下的人类玩家
            if self.game_mode == 'human_vs_ai' and current_player.name[:3] == 'You':
                try:
                    if self.agent:
                        valid_actions = PCplayer.get_valid_actions(current_player.hand, self.last_played)
                        state = self.game_state[self.current_player]
                        action = self.agent.select_action(state, valid_actions)
                        indexes = [current_player.hand.index(card) for card in action]
                        valid, played_cards = self.playing(current_player, indexes)
                        reward = 1.0 if self.check_winner() == self.current_player else 0.0 if valid else -0.5
                        done = self.check_winner() is not None
                        self.agent.store_experience(state, indexes, reward, self.game_state[self.current_player], done)
                    else:
                        selection = input("Enter card indexes to play (space separated) or pass: ")
                    if selection.lower() == 'pass':
                        if not self.last_played: # 作为第一个出牌人，未出牌时不能过
                            print("You must play cards first!")
                            continue
                        else:
                            self.continued_passed_count += 1
                            print('pass')
                            # 记录过牌历史
                            player_index = self.players.index(current_player)
                            self.play_history.append((player_index, []))
                            # 更新游戏状态
                            self.update_game_state()
                            _vail = True  # 设置为有效操作，这样可以继续游戏
                    else:
                        indexes = list(map(int, selection.split()))
                        valid, played_cards = self.playing(current_player, indexes)
                        reward = 1.0 if self.check_winner() == self.current_player else 0.0 if valid else -0.5
                        done = self.check_winner() is not None
                        self.agent.store_experience(state, indexes, reward, self.game_state[self.current_player], done)

                    if _vail:
                        pass
                    else:   
                        continue

                except:
                    print(played_cards)
                    print("Invalid input! Try again.")
                    continue

            # AI玩家（包括AI对战模式下的所有玩家）
            else:
                # 在AI对战模式下添加延迟，使游戏过程可观察
                # if self.game_mode == 'ai_vs_ai':
                    # time.sleep(1)  # 延迟1秒，使AI对战过程更容易观察

                # 电脑AI出牌
                _vail, ai_played_cards = self.ai_player_playing(current_player)

                # 出牌為 None 時視爲過牌
                if ai_played_cards is None:
                    if not self.last_played: # 作为第一个出牌人，未出牌时不能过
                        print(f"{current_player.name} must play cards first!")
                        continue
                    else:
                        print(f'{current_player.name} passes')
                else:
                    if _vail:
                        played_cards = ai_played_cards
                    else:
                        continue
            
            winner = self.check_winner()
            if winner:
                print(f"\n{winner.name} wins!")
                if winner.role == 'landlord':
                    print("Landlord wins!")
                    break
                else:
                    print("Peasants win!")
                    break

            # 下一个玩家
            self.current_player = (self.current_player + 1) % 3 # 下一个玩家
            if self.continued_passed_count == 2:
                self.last_played = []

                # 更新游戏状态
                self.update_game_state()

            current_player = self.players[self.current_player]

            
    def check_winner(self):
        for player in self.players:
            if not player.hand:
                return player
        return None

    def start_game(self):
        self.shuffle_and_deal()
        self.determine_landlord()
        
        # 初始化游戏状态
        self.update_game_state()

        # 游戏开始
        winner = self.check_winner()
        self.play_round(winner)
                

if __name__ == "__main__":
    print("欢迎来到斗地主游戏!")
    print("请选择游戏模式:")
    print("1. 人机对战")
    print("2. AI自我对战")
    
    while True:
        choice = input("请输入选项(1或2): ")
        if choice == '1':
            game_mode = 'human_vs_ai'
            break
        elif choice == '2':
            game_mode = 'ai_vs_ai'
            break
        else:
            print("无效选择，请重新输入!")
    
