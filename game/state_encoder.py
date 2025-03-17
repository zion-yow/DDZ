import numpy as np
from cards import Rank, Suit, Card
import pandas as pd
import engine
import PCplayer

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

    # 打印更详细的游戏状态信息
    print("\n===== 当前游戏状态 =====")
    print(f"当前玩家: {game.players[game.current_player].name}")
    print(f"地主: {next((p.name for p in game.players if p.role == 'landlord'), 'Unknown')}")
    
    # 显示各玩家手牌数量
    print("\n手牌数量:")
    for i, player in enumerate(game.players):
        print(f"{player.name}: {len(player.hand)} 张牌")
    
    # 显示最后出的牌
    print("\n最后出牌:")
    if game.last_played:
        print(game.last_played)
    else:
        print("无")
    
    # 显示历史出牌记录(最近5次)
    print("\n出牌历史(最近5次):")
    recent_history = game.play_history[-5:] if len(game.play_history) >= 5 else game.play_history
    for idx, cards in recent_history:
        player_name = game.players[idx].name
        if cards:
            print(f"{player_name}: {cards}")
        else:
            print(f"{player_name}: 过牌")
    
    # 创建有意义的索引标签
    # 手牌部分 (54维)
    hand_indices = []
    for rank in range(3, 18):  # 3到A，2，小王，大王
        rank_name = get_rank_name(rank)
        if rank <= 15:  # 普通牌
            for suit in ["♠", "♥", "♣", "♦"]:
                hand_indices.append(f"{rank_name}{suit}")
        else:  # 大小王
            hand_indices.append(f"{rank_name}")
    
    # 最后出牌部分 (54维)
    last_played_indices = [f"上家-{idx}" for idx in hand_indices]
    
    # 位置和牌数信息 (3维)
    position_indices = ["地主相对位置", "下家牌数", "上家牌数"]
    
    # 角色信息 (3维)
    role_indices = ["地主", "农民1", "农民2"]
    
    # 历史出牌信息 (前5轮，每个玩家)
    history_indices = []
    for player_idx in range(3):
        player_name = f"玩家{player_idx}"
        for round_idx in range(5):
            for card_idx in hand_indices:
                history_indices.append(f"历史-{player_name}-轮{round_idx}-{card_idx}")
    
    # 合并所有索引
    all_indices = hand_indices + last_played_indices + position_indices + role_indices + history_indices
    
    # 显示状态向量的关键部分
    print("\n状态向量摘要:")
    df = pd.DataFrame(observations, index=all_indices)
    
    # 显示手牌部分
    print("\n手牌部分:")
    print(df.loc[hand_indices].head(10))
    print("...")
    
    # 显示最后出牌部分
    print("\n最后出牌部分:")
    print(df.loc[last_played_indices].head(10))
    print("...")
    
    # 显示位置和角色信息
    print("\n位置和角色信息:")
    print(df.loc[position_indices + role_indices])
    
    # Add action space validation
    valid_actions = [PCplayer.get_valid_actions(p.hand, game.last_played) for p in game.players]
    
    base_state_size = 925  # 54(hand) + 54(last_played) + 3(position) + 3(role) + 810(history) + 1(valid_action)
    return [
        np.concatenate([
            np.array(state_vector, dtype=np.float32),
            np.array([1 if len(valid_actions[i])>0 else 0], dtype=np.float32)
        ]) for i, state_vector in enumerate(observations.values())
    ]

def get_rank_name(rank_value):
    """获取牌面值的可读名称"""
    rank_names = {
        3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10",
        11: "J", 12: "Q", 13: "K", 14: "A", 15: "2", 16: "小王", 17: "大王"
    }
    return rank_names.get(rank_value, str(rank_value))