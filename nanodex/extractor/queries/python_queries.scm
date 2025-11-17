; Function definitions
(function_definition
  name: (identifier) @func.name
  parameters: (parameters) @func.params
  body: (block) @func.body) @func.def

; Class definitions
(class_definition
  name: (identifier) @class.name
  superclasses: (argument_list)? @class.bases
  body: (block) @class.body) @class.def

; Import statements
(import_statement
  name: (dotted_name) @import.name) @import.stmt

(import_from_statement
  module_name: (dotted_name) @import.module
  name: (dotted_name) @import.name) @import.from

; Function calls
(call
  function: [
    (identifier) @call.target
    (attribute) @call.target
  ]
  arguments: (argument_list) @call.args) @call.expr

; Try-except for error handling
(try_statement
  (except_clause
    (as_pattern
      (expression) @error.type
      (as_pattern_target) @error.alias)? @error.handler)? @error.clause) @error.stmt

; Decorators
(decorator
  (identifier) @decorator.name) @decorator

; Variable assignments
(assignment
  left: (identifier) @var.name
  right: (_) @var.value) @var.assign
