/*!
Stack-safe AST traversal via recursion schemes.

Provides base functors ([`AstFrame`], [`ClassSetFrame`]) for use with
[`expand_and_collapse`](recursion::expand_and_collapse).
*/

use alloc::vec::Vec;

use recursion::{MappableFrame, PartiallyApplied};

use crate::ast::{self, Ast, Span};

/// One layer of AST structure with recursive positions replaced by `A`.
///
/// This is the "base functor" for [`Ast`]. Used with [`expand_and_collapse`]
/// from the `recursion` crate for stack-safe traversal.
///
/// [`expand_and_collapse`]: recursion::expand_and_collapse
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub enum AstFrame<'a, A> {
    Empty(Span),
    Flags(&'a ast::SetFlags),
    Literal(&'a ast::Literal),
    Dot(Span),
    Assertion(&'a ast::Assertion),
    ClassUnicode(&'a ast::ClassUnicode),
    ClassPerl(&'a ast::ClassPerl),
    ClassBracketed(&'a ast::ClassBracketed),
    Repetition { span: Span, op: &'a ast::RepetitionOp, greedy: bool, child: A },
    Group { span: Span, kind: &'a ast::GroupKind, child: A },
    Concat { span: Span, children: Vec<A> },
    Alternation { span: Span, children: Vec<A> },
}

impl<'a> MappableFrame for AstFrame<'a, PartiallyApplied> {
    type Frame<X> = AstFrame<'a, X>;

    fn map_frame<A, B>(
        input: Self::Frame<A>,
        mut f: impl FnMut(A) -> B,
    ) -> Self::Frame<B> {
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
            AstFrame::Concat { span, children } => AstFrame::Concat {
                span,
                children: children.into_iter().map(f).collect(),
            },
            AstFrame::Alternation { span, children } => {
                AstFrame::Alternation {
                    span,
                    children: children.into_iter().map(f).collect(),
                }
            }
        }
    }
}

/// Project an AST node into a frame with children as AST references.
///
/// This is the fundamental projection function used by `expand_and_collapse`.
pub fn project_ast(ast: &Ast) -> AstFrame<'_, &Ast> {
    match ast {
        Ast::Empty(span) => AstFrame::Empty(**span),
        Ast::Flags(f) => AstFrame::Flags(f),
        Ast::Literal(lit) => AstFrame::Literal(lit),
        Ast::Dot(span) => AstFrame::Dot(**span),
        Ast::Assertion(a) => AstFrame::Assertion(a),
        Ast::ClassUnicode(c) => AstFrame::ClassUnicode(c),
        Ast::ClassPerl(c) => AstFrame::ClassPerl(c),
        Ast::ClassBracketed(c) => AstFrame::ClassBracketed(c),
        Ast::Repetition(rep) => AstFrame::Repetition {
            span: rep.span,
            op: &rep.op,
            greedy: rep.greedy,
            child: &rep.ast,
        },
        Ast::Group(g) => AstFrame::Group {
            span: g.span,
            kind: &g.kind,
            child: &g.ast,
        },
        Ast::Concat(c) => AstFrame::Concat {
            span: c.span,
            children: c.asts.iter().collect(),
        },
        Ast::Alternation(a) => AstFrame::Alternation {
            span: a.span,
            children: a.asts.iter().collect(),
        },
    }
}

/// One layer of character class structure with recursive positions replaced by `A`.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub enum ClassSetFrame<'a, A> {
    Empty(Span),
    Literal(&'a ast::Literal),
    Range(&'a ast::ClassSetRange),
    Ascii(&'a ast::ClassAscii),
    Unicode(&'a ast::ClassUnicode),
    Perl(&'a ast::ClassPerl),
    Bracketed { span: Span, negated: bool, child: A },
    Union { span: Span, children: Vec<A> },
    BinaryOp { span: Span, kind: ast::ClassSetBinaryOpKind, lhs: A, rhs: A },
}

impl<'a> MappableFrame for ClassSetFrame<'a, PartiallyApplied> {
    type Frame<X> = ClassSetFrame<'a, X>;

    fn map_frame<A, B>(
        input: Self::Frame<A>,
        mut f: impl FnMut(A) -> B,
    ) -> Self::Frame<B> {
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
            ClassSetFrame::Union { span, children } => ClassSetFrame::Union {
                span,
                children: children.into_iter().map(f).collect(),
            },
            ClassSetFrame::BinaryOp { span, kind, lhs, rhs } => {
                ClassSetFrame::BinaryOp {
                    span,
                    kind,
                    lhs: f(lhs),
                    rhs: f(rhs),
                }
            }
        }
    }
}

/// Project a ClassSetItem into a frame.
pub fn project_class_set_item(
    item: &ast::ClassSetItem,
) -> ClassSetFrame<'_, ClassSetChild<'_>> {
    match item {
        ast::ClassSetItem::Empty(span) => ClassSetFrame::Empty(*span),
        ast::ClassSetItem::Literal(lit) => ClassSetFrame::Literal(lit),
        ast::ClassSetItem::Range(r) => ClassSetFrame::Range(r),
        ast::ClassSetItem::Ascii(a) => ClassSetFrame::Ascii(a),
        ast::ClassSetItem::Unicode(u) => ClassSetFrame::Unicode(u),
        ast::ClassSetItem::Perl(p) => ClassSetFrame::Perl(p),
        ast::ClassSetItem::Bracketed(b) => ClassSetFrame::Bracketed {
            span: b.span,
            negated: b.negated,
            child: ClassSetChild::Set(&b.kind),
        },
        ast::ClassSetItem::Union(u) => ClassSetFrame::Union {
            span: u.span,
            children: u.items.iter().map(ClassSetChild::Item).collect(),
        },
    }
}

/// A child in ClassSet traversal (handles mutual recursion between Item and Set).
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub enum ClassSetChild<'a> {
    Item(&'a ast::ClassSetItem),
    Set(&'a ast::ClassSet),
}

/// Project a ClassSetChild into a frame (unifies ClassSetItem and ClassSet).
pub fn project_class_set_child(child: ClassSetChild<'_>) -> ClassSetFrame<'_, ClassSetChild<'_>> {
    match child {
        ClassSetChild::Item(item) => project_class_set_item(item),
        ClassSetChild::Set(set) => match set {
            ast::ClassSet::Item(item) => project_class_set_item(item),
            ast::ClassSet::BinaryOp(op) => ClassSetFrame::BinaryOp {
                span: op.span,
                kind: op.kind,
                lhs: ClassSetChild::Set(&op.lhs),
                rhs: ClassSetChild::Set(&op.rhs),
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::parse::Parser;
    use recursion::{expand_and_collapse, try_expand_and_collapse};

    #[test]
    fn count_nodes() {
        let ast = Parser::new().parse(r"a|b|c").unwrap();
        let count = expand_and_collapse::<AstFrame<PartiallyApplied>, _, _>(
            &ast,
            project_ast,
            |frame| match frame {
                AstFrame::Concat { children, .. }
                | AstFrame::Alternation { children, .. } => {
                    1 + children.into_iter().sum::<usize>()
                }
                AstFrame::Repetition { child, .. }
                | AstFrame::Group { child, .. } => 1 + child,
                _ => 1,
            },
        );
        assert_eq!(count, 4); // 1 alternation + 3 literals
    }

    #[test]
    fn max_depth() {
        let ast = Parser::new().parse(r"((a))").unwrap();
        let depth = expand_and_collapse::<AstFrame<PartiallyApplied>, _, _>(
            &ast,
            project_ast,
            |frame| match frame {
                AstFrame::Group { child, .. }
                | AstFrame::Repetition { child, .. } => 1 + child,
                AstFrame::Concat { children, .. }
                | AstFrame::Alternation { children, .. } => {
                    children.into_iter().max().unwrap_or(0)
                }
                _ => 0,
            },
        );
        assert_eq!(depth, 2);
    }

    #[test]
    fn fallible_traversal() {
        let ast = Parser::new().parse(r"a*").unwrap();
        let result: Result<(), &str> =
            try_expand_and_collapse::<AstFrame<PartiallyApplied>, _, _, _>(
                &ast,
                |node| Ok(project_ast(node)),
                |frame| match frame {
                    AstFrame::Repetition { .. } => Err("no repetitions"),
                    _ => Ok(()),
                },
            );
        assert!(result.is_err());
    }
}
