from collections import Counter


def _is_consecutive(sorted_ranks):
    """检查一组排序后的点数是否连续（如 3-4-5-6-7）"""
    # 排除2和大小王后的有效范围：3到A（3到14）
    return all(r1 + 1 == r2 for r1, r2 in zip(sorted_ranks, sorted_ranks[1:]))

def _exclude_2_and_jokers(ranks):
    return all(r not in [15, 16, 17] for r in ranks)

def get_card_type(cards):
    """判断牌型并返回类型标识和关键值"""
    card_count = len(cards)
    ranks = [c.rank.value for c in cards]
    rank_counter = Counter(ranks)
    unique_ranks = len(rank_counter)
    sorted_ranks = sorted(ranks)

    # 王炸
    if card_count == 2 and {16, 17} == set(ranks):
        return ('rocket', 17)

    # 炸弹（四张）
    if card_count == 4 and unique_ranks == 1:
        return ('bomb', ranks[0])

    # 顺子（至少5张连续单牌，不包含2和大小王）
    if card_count >= 5 and _exclude_2_and_jokers(ranks):
        if unique_ranks == card_count and _is_consecutive(sorted_ranks):
            return ('straight', max(ranks))

    # 连对（至少3对连续，如 33-44-55，不包含2和大小王）
    if card_count >= 6 and card_count % 2 == 0 and _exclude_2_and_jokers(ranks):
        pairs_valid = all(count == 2 for count in rank_counter.values()) # 每个牌面值都是一对
        if pairs_valid and _is_consecutive(sorted_ranks[::2]):
            return ('consec_pairs', max(ranks))
        
    # 飞机，不带翅膀（至少2个连续的三张，如 333-444，不包含2和大小王）
    if card_count >= 6 and card_count % 3 == 0 and _exclude_2_and_jokers(ranks):
        if all(count == 3 for count in rank_counter.values()):
            triple_ranks = sorted(rank_counter.keys())
            if len(triple_ranks) >= 2 and _is_consecutive(triple_ranks):
                return ('plane_without_wing', max(triple_ranks))
    
    # 飞机，带单牌（至少2个连续的三张，如 333-444-5-6，连续牌的飞机不包含2和大小王，且不是两个炸弹）
    if card_count >= 8 and card_count % 4 == 0:
        triple_ranks = [r for r, count in rank_counter.items() if count == 3]
        if ((len(triple_ranks) >= 2 and # 至少两组三张
            _exclude_2_and_jokers(triple_ranks)) and # 三张里没有2和大小王
            any(count != 4 for count in rank_counter.values())): # 不能两个都是炸弹
            sorted_triples = sorted(triple_ranks) # 三张的牌面值排序
            if _is_consecutive(sorted_triples): # 三张的牌面值是连续的
                single_count = sum(1 for count in rank_counter.values() if count == 1) # 单牌数量
                if single_count == len(triple_ranks): # 单牌数量等于三张数量
                    return ('plane_with_single', max(sorted_triples)) # 返回飞机带单牌的类型和最大牌面值

    # 飞机，带对子（至少2个连续的三张，如 333-444-55-66，连续牌的飞机不包含2和大小王，且不是两个炸弹）
    if card_count >= 10 and card_count % 5 == 0:
        triple_ranks = [r for r, count in rank_counter.items() if count == 3]
        if (len(triple_ranks) >= 2 and # 至少两组三张
            _exclude_2_and_jokers(triple_ranks)): # 三张里没有2和大小王
            sorted_triples = sorted(triple_ranks) # 三张的牌面值排序
            if _is_consecutive(sorted_triples): # 三张的牌面值是连续的
                single_count = sum(1 for count in rank_counter.values() if count == 2) # 对子数量
                if single_count == len(triple_ranks): # 对子数量等于三张数量
                    return ('plane_with_duble', max(sorted_triples)) # 返回飞机带对子的类型和最大牌面值

    # 单牌
    if card_count == 1:
        return ('single', ranks[0])

    # 对子
    if card_count == 2 and unique_ranks == 1:
        return ('pair', ranks[0])

    # 三张
    if card_count == 3 and unique_ranks == 1:
        return ('triple', ranks[0])

    # 三带一
    if card_count == 4 and unique_ranks == 2:
        counts = list(rank_counter.values())
        if 3 in counts and 1 in counts:
            three_rank = [r for r, c in rank_counter.items() if c == 3][0]
            return ('triple_with_single', three_rank)
        
    # 三带一对
    if card_count == 5 and unique_ranks == 2:
        counts = list(rank_counter.values())
        if 3 in counts and 2 in counts:
            three_rank = [r for r, c in rank_counter.items() if c == 3][0]
            return ('triple_with_duble', three_rank)


    # 暂时不处理其他复杂牌型
    return (None, 0)

def compare_plays(current_type, last_type):
    """比较两次出牌的大小"""
    if last_type[0] is None:
        return True

    # type_order = ['single', 'pair', 'triple', 'bomb', 'rocket']
    c_type, c_value = current_type
    l_type, l_value = last_type

    # 特殊牌型比较
    if c_type == 'rocket':
        return True  # 王炸最大
    if l_type == 'rocket':
        return False
    
    # 炸弹比较
    if c_type == 'bomb' and l_type != 'bomb':
        return True
    if l_type == 'bomb' and c_type != 'bomb':
        return False
    if c_type == 'bomb' and l_type == 'bomb':
        return c_value > l_value

    # 相同类型比较
    if c_type == l_type:
        return c_value > l_value

    return False  # 类型不同且不是炸弹/王炸

def validate_play(played_cards, last_played):
    """增强的验证逻辑"""
    # 空牌（过牌）处理
    if not played_cards:
        return not bool(last_played)  # 首轮不能过牌

    # 基本牌型验证
    current_type = get_card_type(played_cards)
    if current_type[0] is None:
        return False

    # 首轮出牌，可以出任意合法牌型
    if not last_played:
        return True

    # 比较牌型
    last_type = get_card_type(last_played)
    
    # 类型有效性检查
    if last_type[0] is None:
        return False

    # 牌型必须相同（除非使用炸弹/王炸）
    if current_type[0] not in ['bomb', 'rocket']:
        if current_type[0] != last_type[0]:
            return False

    return compare_plays(current_type, last_type)