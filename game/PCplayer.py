from collections import Counter
import engine as engine

def generate_all_plays(hand):
    sorted_hand = sorted(hand, key=lambda x: x.rank.value)
    plays = {}
    
    # 单牌
    plays['single'] = []
    plays['single'].extend([[card] for card in sorted_hand])

    # 对子
    plays['pair'] = []
    rank_counts = Counter(card.rank.value for card in sorted_hand)
    # 備份對子列表
    pairs_copy = []
    for rank, count in rank_counts.items():
        if count >= 2:
            pair = [c for c in sorted_hand if c.rank.value == rank][:2] # 截取两张牌
            plays['pair'].append(pair)
            pairs_copy.append(pair)

    # 三张
    plays['triple'] = []
    for rank, count in rank_counts.items():
        if count >= 3:
            triple = [c for c in sorted_hand if c.rank.value == rank][:3] # 截取三张牌
            plays['triple'].append(triple)

    # 三带一
    plays['triple_with_single'] = []
    triples = [r for r, c in rank_counts.items() if c >= 3]
    for t_rank in triples:
        triple = [c for c in sorted_hand if c.rank.value == t_rank][:3]
        for c in sorted_hand:
            if c.rank.value != t_rank:
                plays['triple_with_single'].append(triple + [c])

    # 三带一对
    plays['triple_with_pair'] = []
    for t_rank in triples:
        triple = [c for c in sorted_hand if c.rank.value == t_rank][:3]
        pairs = [r for r, c in rank_counts.items() if c >= 2 and r != t_rank]
        for p_rank in pairs:
            pair = [c for c in sorted_hand if c.rank.value == p_rank][:2]
            plays['triple_with_pair'].append(triple +   pair)


    # 飞机（不带翅膀）
    plays['plane_without_wing'] = []
    triple_ranks = sorted([r for r, c in rank_counts.items() if c >= 3])
    for i in range(len(triple_ranks)-1):
        # 判断是否是连续的三张
        if triple_ranks[i+1] == triple_ranks[i] + 1:
            plane = []
            # 三张牌的组合
            for r in triple_ranks[i:i+2]:
                plane.extend([c for c in sorted_hand if c.rank.value == r][:3])
            plays['plane_without_wing'].append(plane)

    # 飞机（带单翅膀），翅膀可以是除了作爲飞机的連續的3*n张牌之外的任意牌
    # 飛機
    plays['plane_with_single'] = []
    plane_played = []
    plane = []
    for i in range(len(triple_ranks)-1):
        if triple_ranks[i+1] == triple_ranks[i] + 1:
            for r in triple_ranks[i:i+2]:
                _plane_1 = [c for c in sorted_hand if c.rank.value == r][:3]
                plane_played.extend(_plane_1)
                plane.extend(_plane_1)
    # 翅膀
    for c in sorted_hand:
        if c.rank.value not in plane_played:
            plane_played.append(c)
            if len(plane) == 8: # 3*n+2
                plays['plane_with_single'].append(plane) 
                break

    # 飞机（带对翅膀），翅膀可以是除了作爲飞机的連續的3*n张牌之外的任意對子
    # 飛機
    plays['plane_with_duble'] = []
    plane_played = []
    plane = []
    for i in range(len(triple_ranks)-1):
        if triple_ranks[i+1] == triple_ranks[i] + 1:
            for r in triple_ranks[i:i+2]:
                _plane_1 = [c for c in sorted_hand if c.rank.value == r][:3]
                plane_played.extend(_plane_1)
                plane.extend(_plane_1)
    # 翅膀
    rank_counts_remained = Counter(card.rank.value for card in [i for i in sorted_hand if i not in plane_played])

    for r, c in rank_counts_remained.items():
        if c >= 2 and r not in plane_played:
            pair = [c for c in sorted_hand if c.rank.value == r][:2]
            plane_played.extend(pair)
            if len(plane) == 10: # 3*n+2
                plays['plane_with_duble'].append(plane)
                break
            
    # 順子
    # 先排除2和王
    plays['straight'] = []
    sorted_hand_ex2_jokers = [c for c in sorted_hand if c.rank.value not in [15,16,17]]
    rank_counts_ex2_jokers = Counter(card.rank.value for card in sorted_hand_ex2_jokers)
    rank_counts_ex2_jokers_keys = sorted(rank_counts_ex2_jokers.keys())

    straight = []
    # 5连以上
    for j in range(4, len(rank_counts_ex2_jokers_keys)):
        for i in range(len(rank_counts_ex2_jokers_keys)-j):
            if rank_counts_ex2_jokers_keys[i] == rank_counts_ex2_jokers_keys[i+j] - j:
                straight = rank_counts_ex2_jokers_keys[i:i+j+1]

                drew_straight = []
                added = []
                for r in sorted_hand_ex2_jokers:
                    if r.rank.value in straight and r.rank.value not in added:
                        drew_straight.append(r)
                        added.append(r.rank.value)

                plays['straight'].append(drew_straight)

    # 連對
    # 已有所有兩張或以上的備份列表：pairs_copy
    # 排除對2
    plays['consec_pairs'] = []
    pairs_copy_ex2 = [c for c in pairs_copy if c[0].rank.value != 15]
    flattened_pair_list = [item for sublist in pairs_copy_ex2 for item in sublist]
    pair_element = flattened_pair_list[::2]

    # 從flattened_pair_list中找出連續的對子
    for j in range(2, len(flattened_pair_list)):
        try:
            for i in range(len(pair_element)):
                if pair_element[i].rank.value == pair_element[i+j].rank.value - j:
                    plays['consec_pairs'].append(flattened_pair_list[i*2:(i+j+1)*2])
        except IndexError:
            break

    # 炸彈
    plays['bomb'] = []
    for rank, count in rank_counts.items():
        if count >= 4:
            triple = [c for c in sorted_hand if c.rank.value == rank][:4] # 截取4张牌
            plays['bomb'].append(triple)
    if len(plays['bomb']) == 0:
        del plays['bomb']
        if_bomb = 0
    else:
        if_bomb = 1

    # 火箭
    plays['rocket'] = []
    draw_jokers = [c for c in sorted_hand if c.rank.value in [16,17]]
    if len(draw_jokers) == 2:
        plays['rocket'].append(draw_jokers)
    if len(plays['rocket']) == 0:
        del plays['rocket']
        if_rocket = 0
    else:
        if_rocket = 1
    
    return plays, if_bomb, if_rocket

def get_valid_actions(hand, last_played):
    # Create index_map to map card object IDs to their indices in hand
    index_map = {id(card): idx for idx, card in enumerate(hand)}
    
    plays, _, _ = generate_all_plays(hand)
    valid_actions = []
    for play_type in plays:
        for play in plays[play_type]:
            if engine.validate_play(play, last_played):
                indexes = [index_map[id(card)] for card in play]
                valid_actions.append(sorted(indexes))
    
    # Remove duplicate combinations
    unique_actions = []
    seen = set()
    for action in valid_actions:
        t = tuple(action)
        if t not in seen:
            seen.add(t)
            unique_actions.append(action)
    return unique_actions

def play(last_play, hand, power=1):
    last_play_type = engine.get_card_type(last_play)
    plays, if_bomb, if_rocket = generate_all_plays(hand)
    print(last_play_type)
    

    if last_play_type[0] is None:
        same_type_hands = []
        for i in plays.values():
            same_type_hands.extend(i)
        print(same_type_hands)
    else:
        try:
            same_type_hands = plays[last_play_type[0]]
        except:
            same_type_hands = []
        if if_bomb > 0:
            same_type_hands.append(plays['bomb']) 

        if if_rocket > 0:
            same_type_hands.append(plays['rocket']) 

    score = 0

    playable = []
    for p in same_type_hands:
        current_play_type = engine.get_card_type(p)
        
        if engine.compare_plays(current_play_type, last_play_type):
            score += 1
            playable.append(p)
        else:
            continue
        if score >= power:
            return p
        else:
            continue
    
    return None

    