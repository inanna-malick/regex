/*!
Provides a stack-safe traversal of the AST using recursion schemes.

This module provides frame types that implement `MappableFrame` from the
`recursion` crate, enabling stack-safe traversal via `collapse_frames`.

Key types:
- [`AstFrame`]: One layer of AST with child positions replaced by a type parameter
- [`ClassSetFrame`]: One layer of character class structure

The frame types can be used directly with the recursion crate's
`expand_and_collapse` for custom traversals that thread context (like flags)
through the tree.
*/

use alloc::vec::Vec;

use recursion::{MappableFrame, PartiallyApplied};

use crate::ast::{self, Ast, Span};

// Re-export recursion crate primitives for use by consumers
pub use recursion::{Collapsible, CollapsibleExt, Expandable, ExpandableExt};

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

/// Project an AST node into a frame with children as AST references.
///
/// This is the fundamental projection function used by `expand_and_collapse`.
pub fn project_ast(ast: &Ast) -> AstFrame<&Ast> {
    match ast {
        Ast::Empty(span) => AstFrame::Empty(**span),
        Ast::Flags(f) => AstFrame::Flags((**f).clone()),
        Ast::Literal(lit) => AstFrame::Literal((**lit).clone()),
        Ast::Dot(span) => AstFrame::Dot(**span),
        Ast::Assertion(a) => AstFrame::Assertion((**a).clone()),
        Ast::ClassUnicode(c) => AstFrame::ClassUnicode((**c).clone()),
        Ast::ClassPerl(c) => AstFrame::ClassPerl((**c).clone()),
        Ast::ClassBracketed(c) => AstFrame::ClassBracketed((**c).clone()),
        Ast::Repetition(rep) => AstFrame::Repetition {
            span: rep.span,
            op: rep.op.clone(),
            greedy: rep.greedy,
            child: &rep.ast,
        },
        Ast::Group(g) => AstFrame::Group {
            span: g.span,
            kind: g.kind.clone(),
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
        /// Span of the bracketed class.
        span: Span,
        /// Whether this class is negated.
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

/// Project a ClassSet into a frame.
pub fn project_class_set(set: &ast::ClassSet) -> ClassSetFrame<&ast::ClassSet> {
    match set {
        ast::ClassSet::Item(item) => project_class_set_item_as_set(item),
        ast::ClassSet::BinaryOp(op) => ClassSetFrame::BinaryOp {
            span: op.span,
            kind: op.kind,
            lhs: &op.lhs,
            rhs: &op.rhs,
        },
    }
}

fn project_class_set_item_as_set(item: &ast::ClassSetItem) -> ClassSetFrame<&ast::ClassSet> {
    // For items that aren't naturally ClassSet, we need to handle them specially.
    // This is a bit awkward because ClassSetItem::Union contains ClassSetItems, not ClassSets.
    // For now, we'll panic on Union - callers should use project_class_set_item instead.
    match item {
        ast::ClassSetItem::Empty(span) => ClassSetFrame::Empty(*span),
        ast::ClassSetItem::Literal(lit) => ClassSetFrame::Literal(lit.clone()),
        ast::ClassSetItem::Range(r) => ClassSetFrame::Range(r.clone()),
        ast::ClassSetItem::Ascii(a) => ClassSetFrame::Ascii(a.clone()),
        ast::ClassSetItem::Unicode(u) => ClassSetFrame::Unicode(u.clone()),
        ast::ClassSetItem::Perl(p) => ClassSetFrame::Perl(p.clone()),
        ast::ClassSetItem::Bracketed(b) => ClassSetFrame::Bracketed {
            span: b.span,
            negated: b.negated,
            child: &b.kind,
        },
        ast::ClassSetItem::Union(_) => {
            panic!("project_class_set_item_as_set called on Union - use dedicated traversal")
        }
    }
}

/// Project a ClassSetItem into a frame with ClassSetItem children.
pub fn project_class_set_item(item: &ast::ClassSetItem) -> ClassSetFrame<ClassSetChild<'_>> {
    match item {
        ast::ClassSetItem::Empty(span) => ClassSetFrame::Empty(*span),
        ast::ClassSetItem::Literal(lit) => ClassSetFrame::Literal(lit.clone()),
        ast::ClassSetItem::Range(r) => ClassSetFrame::Range(r.clone()),
        ast::ClassSetItem::Ascii(a) => ClassSetFrame::Ascii(a.clone()),
        ast::ClassSetItem::Unicode(u) => ClassSetFrame::Unicode(u.clone()),
        ast::ClassSetItem::Perl(p) => ClassSetFrame::Perl(p.clone()),
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

/// A child reference in a ClassSet traversal - either an Item or a Set.
#[derive(Clone, Debug)]
pub enum ClassSetChild<'a> {
    /// A ClassSetItem child.
    Item(&'a ast::ClassSetItem),
    /// A ClassSet child.
    Set(&'a ast::ClassSet),
}

/// Frame wrapper that pairs a frame with context (like flags).
///
/// This enables the "flags in seed" pattern: during expansion, context flows
/// down to children; during collapse, each frame has its context available.
#[derive(Clone, Debug)]
pub struct WithContext<F, C> {
    /// The frame being wrapped.
    pub frame: F,
    /// The context associated with this frame.
    pub context: C,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::parse::Parser;
    use recursion::expand_and_collapse;

    #[test]
    fn test_count_nodes() {
        let ast = Parser::new().parse(r"a|b|c").unwrap();

        let count = expand_and_collapse::<AstFrame<PartiallyApplied>, _, _>(
            &ast,
            |node| project_ast(node),
            |frame| match frame {
                AstFrame::Concat { children, .. } => 1 + children.into_iter().sum::<usize>(),
                AstFrame::Alternation { children, .. } => 1 + children.into_iter().sum::<usize>(),
                AstFrame::Repetition { child, .. } => 1 + child,
                AstFrame::Group { child, .. } => 1 + child,
                _ => 1,
            },
        );
        // a|b|c parses to Alternation with 3 Literal children
        assert_eq!(count, 4); // 1 alternation + 3 literals
    }

    #[test]
    fn test_depth() {
        let ast = Parser::new().parse(r"((a))").unwrap();

        let depth = expand_and_collapse::<AstFrame<PartiallyApplied>, _, _>(
            &ast,
            |node| project_ast(node),
            |frame| match frame {
                AstFrame::Group { child, .. } => 1 + child,
                AstFrame::Repetition { child, .. } => 1 + child,
                AstFrame::Concat { children, .. } => {
                    children.into_iter().max().unwrap_or(0)
                }
                AstFrame::Alternation { children, .. } => {
                    children.into_iter().max().unwrap_or(0)
                }
                _ => 0,
            },
        );
        assert_eq!(depth, 2); // Two nested groups
    }

    #[test]
    fn test_fallible() {
        let ast = Parser::new().parse(r"a*").unwrap();

        let result: Result<(), &str> = recursion::try_expand_and_collapse::<AstFrame<PartiallyApplied>, _, _, _>(
            &ast,
            |node| Ok(project_ast(node)),
            |frame| match frame {
                AstFrame::Repetition { .. } => Err("no repetitions allowed"),
                _ => Ok(()),
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_with_context() {
        // Demonstrate threading context through traversal
        let ast = Parser::new().parse(r"(?i)a").unwrap();

        #[derive(Clone, Debug, Default)]
        struct Ctx { case_insensitive: bool }

        // Count nodes, tracking if we're inside a case-insensitive group
        let (count, _) = expand_and_collapse::<AstFrame<PartiallyApplied>, _, _>(
            (&ast, Ctx::default()),
            |(node, ctx)| {
                let frame = project_ast(node);
                // If this is a group with flags, update context for children
                let child_ctx = match &frame {
                    AstFrame::Group { kind: ast::GroupKind::NonCapturing(flags), .. } => {
                        let mut new_ctx = ctx.clone();
                        for item in &flags.items {
                            if let ast::FlagsItemKind::Flag(ast::Flag::CaseInsensitive) = item.kind {
                                new_ctx.case_insensitive = true;
                            }
                        }
                        new_ctx
                    }
                    _ => ctx.clone(),
                };
                AstFrame::map_frame(frame, |child| (child, child_ctx.clone()))
            },
            |frame| match frame {
                AstFrame::Concat { children, .. } |
                AstFrame::Alternation { children, .. } => {
                    let sum: usize = children.iter().map(|(c, _)| c).sum();
                    let ctx = children.into_iter().next().map(|(_, c)| c).unwrap_or_default();
                    (1 + sum, ctx)
                }
                AstFrame::Repetition { child: (c, ctx), .. } |
                AstFrame::Group { child: (c, ctx), .. } => (1 + c, ctx),
                _ => (1, Ctx::default()),
            },
        );
        assert!(count >= 2); // At least the group and literal
    }
}
