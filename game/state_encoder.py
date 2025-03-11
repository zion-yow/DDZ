import numpy as np
from cards import Rank, Suit, Card
import pandas as pd

# 牌面值映射
CARD_RANK_TO_ID = {
    3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
    11: 8, 12: 9, 13: 10, 14: 11, 15: 12, 16: 13, 17: 14
}

# 花色映射
CARD_SUIT_TO_ID = {
    "♠": 0, "♥": 1, "♣": 2, "♦": 3, "JOKER": 4
}

# 角色映射
ROLE_ID = {
    'landlord': 0,
    'peasant1': 1,
    'peasant2': 2
}

def get_card_index(card):
    """获取卡牌的唯一索引"""
    rank_id = CARD_RANK_TO_ID[card.rank.value]
    suit_id = CARD_SUIT_TO_ID[card.suit.value]
    
    # 大小王特殊处理
    if card.rank in [Rank.SMALL_JOKER, Rank.BIG_JOKER]:
        return 52 + (1 if card.rank == Rank.BIG_JOKER else 0)
    
    # 普通牌
    return rank_id * 4 + suit_id

def cards_to_vector(cards):
    """将卡牌列表转换为one-hot向量"""
    vector = np.zeros(54, dtype=np.int8)
    for card in cards:
        vector[get_card_index(card)] = 1
    return vector

def encode_state(game, player_position):
    """
    编码游戏状态
    :param game: DouDiZhuGame实例
    :param player_position: 当前玩家位置(0,1,2)
    :return: 状态向量
    """
    current_player = game.players[player_position]
    
    # 基础信息
    landlord_position = next((i for i, p in enumerate(game.players) if p.role == 'landlord'), 0)
    relative_position = (landlord_position - player_position) % 3
    
    # 手牌编码 (54维)
    my_cards = cards_to_vector(current_player.hand)
    
    # 其他玩家的相对位置
    next_player_position = (player_position + 1) % 3
    next_next_player_position = (player_position + 2) % 3
    
    # 其他玩家手牌数量
    next_player_num_cards = len(game.players[next_player_position].hand)
    next_next_player_num_cards = len(game.players[next_next_player_position].hand)
    
    # 最后出牌编码 (54维)
    last_played_cards = cards_to_vector(game.last_played)
    
    # 历史出牌信息 (前五轮)
    history_plays = []
    if hasattr(game, 'play_history'):
        # 获取最近五轮的出牌历史
        recent_history = game.play_history[-15:] if len(game.play_history) > 15 else game.play_history
        # 每个玩家最多取最近5次出牌
        for i in range(3):
            player_history = [cards for player_idx, cards in recent_history if player_idx == i][-5:] if recent_history else []
            # 如果不足5次，用空列表补齐
            while len(player_history) < 5:
                player_history.append([])
            # 将每次出牌转换为向量并添加到历史中
            for cards in player_history:
                history_plays.append(cards_to_vector(cards))
    else:
        # 如果没有历史记录，用15个空向量代替(3个玩家各5轮)
        for _ in range(15):
            history_plays.append(np.zeros(54, dtype=np.int8))
    
    # 将历史出牌向量连接起来
    history_plays_vector = np.concatenate(history_plays)
    
    # 角色信息 (3维one-hot)
    role_info = np.zeros(3, dtype=np.int8)
    role_info[ROLE_ID[current_player.role] if current_player.role else 0] = 1
    
    # 合并所有特征
    state = np.concatenate([
        my_cards,                      # 我的手牌 (54维)
        last_played_cards,             # 最后出的牌 (54维)
        np.array([relative_position, next_player_num_cards, next_next_player_num_cards], dtype=np.int8),  # 位置和牌数信息 (3维)
        role_info,                     # 角色信息 (3维)
        history_plays_vector           # 历史出牌信息 (54*15=810维)
    ])
    
    return state

def get_obs(game):
    """
    获取所有玩家的观察状态
    :param game: DouDiZhuGame实例
    :return: 三个玩家的状态向量
    """
    observations = {}
    for i in range(3):
        observations[i] = encode_state(game, i)

    print(pd.DataFrame(observations))
    return observations