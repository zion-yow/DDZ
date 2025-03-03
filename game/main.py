import random
from enum import Enum
from collections import Counter
from itertools import groupby
from cards import Suit, Rank, Card, Player
import engine as engine
import PCplayer as PCplayer




class DouDiZhuGame:
    def __init__(self):
        self.players = [Player("You",''), Player("PCPlayer1", 'peasant1'), Player("PCPlayer2", 'peasant2')]
        self.deck = [] # 牌堆
        self.current_player = 0
        self.last_played = [] # 上家出的牌
        self.landlord_cards = []
        self.continued_passed_count = 0
        self.create_deck()

    
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


    def playing(self, current_player, cardindexes):
        """出牌"""
        indexes = cardindexes.copy()
        played_cards = [current_player.hand[i] for i in indexes] # 出的牌
        valider = engine.validate_play(played_cards, self.last_played)

        if valider:
            # 移除手牌
            print(current_player.hand)
            for i in sorted(indexes, reverse=True):
                del current_player.hand[i]
            self.last_played = played_cards
            
            #  连续过牌次数清零
            self.continued_passed_count = 0
            # 展示出的牌
            print(played_cards)
            return True, played_cards

        else:
            print(played_cards)
            print("Invalid play! Try again.")

            return False, played_cards

    def ai_player_playing(self, current_player):
        """ai的选牌"""
        played = PCplayer.play(self.last_played, current_player.hand, power=1)
        valider = engine.validate_play(played, self.last_played)
        
        if valider:
            # 移除手牌
            current_player.hand.remove(played[0])
            self.last_played = played

            if played == []:
                return False, played
            else:
                #  连续过牌次数清零
                self.continued_passed_count = 0
                # 展示出的牌
                print(played)
                return True, played

        else:
            print(played)
            print("Invalid play! Try again.")
            return False, played

    def play_round(self, winner):
        """一轮游戏"""
        current_player = self.players[self.current_player]
        valid_play = False

        while not valid_play and not winner:
            print(current_player.name)
            print(f"\n{current_player.name}'s turn")
            print( current_player.hand) # 假设AI明牌
            # 玩家出牌
            if current_player.name[:3] == 'You':
                try:
                    selection = input("Enter card indexes to play (space separated) or pass: ")# 输入要出的牌的索引，用空格分隔
                    if selection.lower() == 'pass':
                        self.continued_passed_count += 1
                        if not self.last_played: # 作为第一个出牌人，未出牌时不能过
                            print("You must play cards first!")
                            continue
                        else:
                            print('pass')
                            _vail = True # 讓過
                    else:
                        indexes = list(map(int, selection.split()))
                        _vail, played_cards = self.playing(current_player, indexes)

                    if _vail:
                        pass
                    else:   
                        continue

                except:
                    print(played_cards)
                    print("Invalid input! Try again.")
                    continue

            # 电脑出牌
            else:
                # 电脑AI出牌

                '''AI出牌逻辑'''

                _vail, ai_played_cards = self.ai_player_playing(current_player)

                # 出牌為 None 時視爲過牌
                if ai_played_cards is None:
                    if not self.last_played: # 作为第一个出牌人，未出牌时不能过
                        print("You must play cards first!")
                        continue
                    else:
                        self.continued_passed_count += 1
                        print('pass')
                else:
                    if _vail:
                        played_cards = ai_played_cards
                        pass
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
            current_player = self.players[self.current_player]

            
                    

    def check_winner(self):
        for player in self.players:
            if not player.hand:
                return player
        return None

    def start_game(self):
        self.shuffle_and_deal()
        self.determine_landlord()
        
        # 游戏开始
        winner = self.check_winner()
        self.play_round(winner)
                

if __name__ == "__main__":
    game = DouDiZhuGame()
    game.start_game()
