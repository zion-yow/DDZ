from enum import Enum
from collections import Counter

class Suit(Enum):
    SPADE = "♠"
    HEART = "♥"
    CLUB = "♣"
    DIAMOND = "♦"
    JOKER = "JOKER"

class Rank(Enum):
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    J = 11
    Q = 12
    K = 13
    A = 14
    TWO = 15
    SMALL_JOKER = 16
    BIG_JOKER = 17

class Card:
    __slots__ = ('suit', 'rank')  # 优化内存使用

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __repr__(self):
        if self.rank in {Rank.SMALL_JOKER, Rank.BIG_JOKER}:
            return self.rank.name.replace('_', ' ')
        return f"{self.suit.value}{self.rank.value}"

class Player:
    def __init__(self, name, role):
        self.name = name
        self.hand = []
        self.role = role  # 'peasant1', 'peasant2'

    def sort_hand(self):
        self.hand.sort(key=lambda x: (x.rank.value, x.suit.value))