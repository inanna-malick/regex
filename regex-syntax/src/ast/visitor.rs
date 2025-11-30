/*!
Provides a stack-safe traversal of the AST using recursion schemes.

This module replaces the hand-rolled stack machine with the `recursion` crate's
`Collapsible` pattern. The key types are:

- [`AstFrame`]: One layer of AST structure with child positions replaced by a type parameter
- [`ClassSetFrame`]: One layer of character class structure

These frame types implement `MappableFrame` from the recursion crate, enabling
stack-safe traversal via `collapse_frames` and `try_collapse_frames`.
*/

use alloc::vec::Vec;

use recursion::{Collapsible, CollapsibleExt, MappableFrame, PartiallyApplied};

use crate::ast::{self, Ast, Span};

/// A single layer of AST structure with recursive positions replaced by `A`.
///
/// This is the "base functor" for the AST type. Each variant corresponds to an
/// AST node, but recursive child positions contain `A` instead of `Box<Ast>`.
#[derive(Clone, Debug)]
pub enum AstFrame<A> {
    /// An empty regex (matches the empty string).
    Empty(Span),
    /// A set of flags, e.g., `(?is)`.
    Flags(ast::SetFlags),
    /// A literal character or escape sequence.
    Literal(ast::Literal),
    /// The "any character" class, e.g., `.`.
    Dot(Span),
    /// An assertion (anchor), e.g., `^`, `$`, `\b`.
    Assertion(ast::Assertion),
    /// A Unicode character class, e.g., `\pN`.
    ClassUnicode(ast::ClassUnicode),
    /// A Perl character class, e.g., `\d`, `\s`, `\w`.
    ClassPerl(ast::ClassPerl),
    /// A bracketed character class, e.g., `[a-z]`.
    ClassBracketed(ast::ClassBracketed),
    /// A repetition, e.g., `a*`, `a+`, `a?`, `a{1,3}`.
    Repetition {
        /// Span of the repetition.
        span: Span,
        /// The repetition operator.
        op: ast::RepetitionOp,
        /// Whether the repetition is greedy.
        greedy: bool,
        /// The sub-expression being repeated (already collapsed).
        child: A,
    },
    /// A grouped sub-expression, e.g., `(a)`, `(?:a)`, `(?P<name>a)`.
    Group {
        /// Span of the group.
        span: Span,
        /// The kind of group.
        kind: ast::GroupKind,
        /// The sub-expression inside the group (already collapsed).
        child: A,
    },
    /// A concatenation of sub-expressions.
    Concat {
        /// Span of the concatenation.
        span: Span,
        /// The sub-expressions (already collapsed, in order).
        children: Vec<A>,
    },
    /// An alternation of sub-expressions, e.g., `a|b|c`.
    Alternation {
        /// Span of the alternation.
        span: Span,
        /// The alternative branches (already collapsed, in order).
        children: Vec<A>,
    },
}

impl MappableFrame for AstFrame<PartiallyApplied> {
    type Frame<X> = AstFrame<X>;

    fn map_frame<A, B>(input: Self::Frame<A>, mut f: impl FnMut(A) -> B) -> Self::Frame<B> {
        match input {
            AstFrame::Empty(span) => AstFrame::Empty(span),
            AstFrame::Flags(flags) => AstFrame::Flags(flags),
            AstFrame::Literal(lit) => AstFrame::Literal(lit),
            AstFrame::Dot(span) => AstFrame::Dot(span),
            AstFrame::Assertion(a) => AstFrame::Assertion(a),
            AstFrame::ClassUnicode(c) => AstFrame::ClassUnicode(c),
            AstFrame::ClassPerl(c) => AstFrame::ClassPerl(c),
            AstFrame::ClassBracketed(c) => AstFrame::ClassBracketed(c),
            AstFrame::Repetition { span, op, greedy, child } => {
                AstFrame::Repetition { span, op, greedy, child: f(child) }
            }
            AstFrame::Group { span, kind, child } => {
                AstFrame::Group { span, kind, child: f(child) }
            }
            AstFrame::Concat { span, children } => {
                AstFrame::Concat { span, children: children.into_iter().map(f).collect() }
            }
            AstFrame::Alternation { span, children } => {
                AstFrame::Alternation { span, children: children.into_iter().map(f).collect() }
            }
        }
    }
}

/// Owned AST wrapper for Collapsible implementation.
///
/// The recursion crate requires owned values, so we clone the AST.
/// For large ASTs, consider using a reference-based approach with
/// explicit stack management instead.
#[derive(Clone, Debug)]
pub struct OwnedAst(pub Ast);

impl Collapsible for OwnedAst {
    type FrameToken = AstFrame<PartiallyApplied>;

    fn into_frame(self) -> AstFrame<Self> {
        match self.0 {
            Ast::Empty(span) => AstFrame::Empty(*span),
            Ast::Flags(f) => AstFrame::Flags(*f),
            Ast::Literal(lit) => AstFrame::Literal(*lit),
            Ast::Dot(span) => AstFrame::Dot(*span),
            Ast::Assertion(a) => AstFrame::Assertion(*a),
            Ast::ClassUnicode(c) => AstFrame::ClassUnicode(*c),
            Ast::ClassPerl(c) => AstFrame::ClassPerl(*c),
            Ast::ClassBracketed(c) => AstFrame::ClassBracketed(*c),
            Ast::Repetition(rep) => AstFrame::Repetition {
                span: rep.span,
                op: rep.op.clone(),
                greedy: rep.greedy,
                child: OwnedAst((*rep.ast).clone()),
            },
            Ast::Group(g) => AstFrame::Group {
                span: g.span,
                kind: g.kind.clone(),
                child: OwnedAst((*g.ast).clone()),
            },
            Ast::Concat(c) => AstFrame::Concat {
                span: c.span,
                children: c.asts.into_iter().map(OwnedAst).collect(),
            },
            Ast::Alternation(a) => AstFrame::Alternation {
                span: a.span,
                children: a.asts.into_iter().map(OwnedAst).collect(),
            },
        }
    }
}

/// A single layer of character class set structure.
#[derive(Clone, Debug)]
pub enum ClassSetFrame<A> {
    /// An empty class set item.
    Empty(Span),
    /// A literal in a class.
    Literal(ast::Literal),
    /// A range in a class, e.g., `a-z`.
    Range(ast::ClassSetRange),
    /// An ASCII class, e.g., `[:alpha:]`.
    Ascii(ast::ClassAscii),
    /// A Unicode class inside a bracketed class.
    Unicode(ast::ClassUnicode),
    /// A Perl class inside a bracketed class.
    Perl(ast::ClassPerl),
    /// A nested bracketed class.
    Bracketed {
        /// Span and negation info.
        span: Span,
        negated: bool,
        /// The contents of the nested class (already collapsed).
        child: A,
    },
    /// A union of class items.
    Union {
        /// Span of the union.
        span: Span,
        /// The items in the union (already collapsed).
        children: Vec<A>,
    },
    /// A binary set operation (intersection, difference, symmetric difference).
    BinaryOp {
        /// Span of the operation.
        span: Span,
        /// The kind of operation.
        kind: ast::ClassSetBinaryOpKind,
        /// Left-hand side (already collapsed).
        lhs: A,
        /// Right-hand side (already collapsed).
        rhs: A,
    },
}

impl MappableFrame for ClassSetFrame<PartiallyApplied> {
    type Frame<X> = ClassSetFrame<X>;

    fn map_frame<A, B>(input: Self::Frame<A>, mut f: impl FnMut(A) -> B) -> Self::Frame<B> {
        match input {
            ClassSetFrame::Empty(span) => ClassSetFrame::Empty(span),
            ClassSetFrame::Literal(lit) => ClassSetFrame::Literal(lit),
            ClassSetFrame::Range(r) => ClassSetFrame::Range(r),
            ClassSetFrame::Ascii(a) => ClassSetFrame::Ascii(a),
            ClassSetFrame::Unicode(u) => ClassSetFrame::Unicode(u),
            ClassSetFrame::Perl(p) => ClassSetFrame::Perl(p),
            ClassSetFrame::Bracketed { span, negated, child } => {
                ClassSetFrame::Bracketed { span, negated, child: f(child) }
            }
            ClassSetFrame::Union { span, children } => {
                ClassSetFrame::Union { span, children: children.into_iter().map(f).collect() }
            }
            ClassSetFrame::BinaryOp { span, kind, lhs, rhs } => {
                ClassSetFrame::BinaryOp { span, kind, lhs: f(lhs), rhs: f(rhs) }
            }
        }
    }
}

/// Owned ClassSet wrapper for Collapsible implementation.
#[derive(Clone, Debug)]
pub struct OwnedClassSet(pub ast::ClassSet);

impl Collapsible for OwnedClassSet {
    type FrameToken = ClassSetFrame<PartiallyApplied>;

    fn into_frame(self) -> ClassSetFrame<Self> {
        match self.0 {
            ast::ClassSet::Item(item) => class_set_item_to_frame(item),
            ast::ClassSet::BinaryOp(op) => ClassSetFrame::BinaryOp {
                span: op.span,
                kind: op.kind,
                lhs: OwnedClassSet(*op.lhs),
                rhs: OwnedClassSet(*op.rhs),
            },
        }
    }
}

fn class_set_item_to_frame(item: ast::ClassSetItem) -> ClassSetFrame<OwnedClassSet> {
    match item {
        ast::ClassSetItem::Empty(span) => ClassSetFrame::Empty(span),
        ast::ClassSetItem::Literal(lit) => ClassSetFrame::Literal(lit),
        ast::ClassSetItem::Range(r) => ClassSetFrame::Range(r),
        ast::ClassSetItem::Ascii(a) => ClassSetFrame::Ascii(a),
        ast::ClassSetItem::Unicode(u) => ClassSetFrame::Unicode(u),
        ast::ClassSetItem::Perl(p) => ClassSetFrame::Perl(p),
        ast::ClassSetItem::Bracketed(b) => ClassSetFrame::Bracketed {
            span: b.span,
            negated: b.negated,
            child: OwnedClassSet(b.kind),
        },
        ast::ClassSetItem::Union(u) => ClassSetFrame::Union {
            span: u.span,
            children: u.items.into_iter().map(|i| OwnedClassSet(ast::ClassSet::Item(i))).collect(),
        },
    }
}

/// Collapse an AST using a provided algebra function.
///
/// This is the primary way to process an AST with the recursion crate.
/// The algebra receives each node after its children have been processed,
/// with children replaced by the results of processing them.
///
/// # Stack Safety
///
/// This function is stack-safe: it uses heap allocation proportional to the
/// AST size rather than stack space proportional to AST depth.
pub fn collapse_ast<Out>(
    ast: Ast,
    f: impl FnMut(AstFrame<Out>) -> Out,
) -> Out {
    OwnedAst(ast).collapse_frames(f)
}

/// Collapse an AST using a fallible algebra function.
///
/// Like [`collapse_ast`] but the algebra can return errors, which will
/// short-circuit the traversal.
pub fn try_collapse_ast<Out, E>(
    ast: Ast,
    f: impl FnMut(AstFrame<Out>) -> Result<Out, E>,
) -> Result<Out, E> {
    OwnedAst(ast).try_collapse_frames(f)
}

/// Collapse a ClassSet using a provided algebra function.
pub fn collapse_class_set<Out>(
    set: ast::ClassSet,
    f: impl FnMut(ClassSetFrame<Out>) -> Out,
) -> Out {
    OwnedClassSet(set).collapse_frames(f)
}

/// Collapse a ClassSet using a fallible algebra function.
pub fn try_collapse_class_set<Out, E>(
    set: ast::ClassSet,
    f: impl FnMut(ClassSetFrame<Out>) -> Result<Out, E>,
) -> Result<Out, E> {
    OwnedClassSet(set).try_collapse_frames(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::parse::Parser;

    #[test]
    fn test_count_nodes() {
        let ast = Parser::new().parse(r"a|b|c").unwrap();
        let count = collapse_ast(ast, |frame| match frame {
            AstFrame::Concat { children, .. } => 1 + children.into_iter().sum::<usize>(),
            AstFrame::Alternation { children, .. } => 1 + children.into_iter().sum::<usize>(),
            AstFrame::Repetition { child, .. } => 1 + child,
            AstFrame::Group { child, .. } => 1 + child,
            _ => 1,
        });
        // a|b|c parses to Alternation with 3 Literal children
        assert_eq!(count, 4); // 1 alternation + 3 literals
    }

    #[test]
    fn test_depth() {
        let ast = Parser::new().parse(r"((a))").unwrap();
        let depth = collapse_ast(ast, |frame| match frame {
            AstFrame::Group { child, .. } => 1 + child,
            AstFrame::Repetition { child, .. } => 1 + child,
            AstFrame::Concat { children, .. } => {
                children.into_iter().max().unwrap_or(0)
            }
            AstFrame::Alternation { children, .. } => {
                children.into_iter().max().unwrap_or(0)
            }
            _ => 0,
        });
        assert_eq!(depth, 2); // Two nested groups
    }

    #[test]
    fn test_fallible() {
        let ast = Parser::new().parse(r"a*").unwrap();
        let result: Result<(), &str> = try_collapse_ast(ast, |frame| match frame {
            AstFrame::Repetition { .. } => Err("no repetitions allowed"),
            _ => Ok(()),
        });
        assert!(result.is_err());
    }
}
