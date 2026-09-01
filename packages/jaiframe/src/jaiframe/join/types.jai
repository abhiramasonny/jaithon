#: The vocabulary a caller spells a join with: `How`, `Validate`, and the
#: direction an asof merge looks in.
#:
#: Each one carries pandas' spelling both ways, so an argument written as a
#: string in pandas reads the same here.

#: How the rows of two frames are paired up.
pub enum How {
    #: Only the rows whose keys appear on both sides.
    Inner,
    #: Every left row, with nulls where the right side has none.
    Left,
    #: Every right row, with nulls where the left side has none.
    Right,
    #: Every row of both sides.
    Outer,
    #: Every left row against every right row, on no key at all.
    Cross,

    #: The pandas spelling.
    pub fn name(self) -> str {
        return match self {
            How.Inner => "inner",
            How.Left => "left",
            How.Right => "right",
            How.Outer => "outer",
            How.Cross => "cross",
        }
    }

    #: The join `name` spells, so `merge(how: How.from_name("left"))` reads.
    pub static fn from_name(text: str) -> How {
        return match text {
            "inner" => How.Inner,
            "left" => How.Left,
            "right" => How.Right,
            "outer" | "full" => How.Outer,
            "cross" => How.Cross,
            _ => throw ValueError(f"unknown join {text}"),
        }
    }

    #: Whether a left row with no match still appears.
    pub fn keeps_left(self) -> bool {
        return match self {
            How.Left | How.Outer => true,
            _ => false,
        }
    }

    #: Whether a right row with no match still appears.
    pub fn keeps_right(self) -> bool {
        return match self {
            How.Right | How.Outer => true,
            _ => false,
        }
    }
}

#: The shape a merge's keys are promised to have.
#:
#: Checked over the keys that take part in the join rather than over the whole
#: frame, which is where a duplicate would change the answer: a key present on
#: only one side cannot multiply anything.
pub enum Validate {
    #: Every key at most once on each side.
    OneToOne,
    #: Every key at most once on the left, any number on the right.
    OneToMany,
    #: Any number on the left, at most once on the right.
    ManyToOne,
    #: No promise; the check is skipped.
    ManyToMany,

    #: The pandas spelling.
    pub fn name(self) -> str {
        return match self {
            Validate.OneToOne => "1:1",
            Validate.OneToMany => "1:m",
            Validate.ManyToOne => "m:1",
            Validate.ManyToMany => "m:m",
        }
    }

    #: The promise `name` spells.
    pub static fn from_name(text: str) -> Validate {
        return match text {
            "1:1" | "one_to_one" => Validate.OneToOne,
            "1:m" | "one_to_many" => Validate.OneToMany,
            "m:1" | "many_to_one" => Validate.ManyToOne,
            "m:m" | "many_to_many" => Validate.ManyToMany,
            _ => throw ValueError(f"unknown validation {text}"),
        }
    }

    #: Whether the left side promises unique keys.
    pub fn left_unique(self) -> bool {
        return match self {
            Validate.OneToOne | Validate.OneToMany => true,
            _ => false,
        }
    }

    #: Whether the right side promises unique keys.
    pub fn right_unique(self) -> bool {
        return match self {
            Validate.OneToOne | Validate.ManyToOne => true,
            _ => false,
        }
    }
}

#: Which way `merge_asof` looks when a key has no exact match.
pub enum AsofDirection {
    #: The last right row at or before the left key.
    Backward,
    #: The first right row at or after the left key.
    Forward,
    #: Whichever of those two is closer; a tie goes backward.
    Nearest,

    #: The pandas spelling.
    pub fn name(self) -> str {
        return match self {
            AsofDirection.Backward => "backward",
            AsofDirection.Forward => "forward",
            AsofDirection.Nearest => "nearest",
        }
    }

    #: The direction `name` spells.
    pub static fn from_name(text: str) -> AsofDirection {
        return match text {
            "backward" => AsofDirection.Backward,
            "forward" => AsofDirection.Forward,
            "nearest" => AsofDirection.Nearest,
            _ => throw ValueError(f"unknown asof direction {text}"),
        }
    }
}
