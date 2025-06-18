
import 'package:flutter/material.dart';

class CardContainer extends StatelessWidget {

  final Widget child;
  final double? width;

  const CardContainer({super.key, required this.child, this.width});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: width ?? 600,
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: const [
          BoxShadow(
            color: Colors.black54,
            blurRadius: 15,
            offset: Offset(0, 4),
          )
        ],
      ),
      child: child,
    );
  }
}