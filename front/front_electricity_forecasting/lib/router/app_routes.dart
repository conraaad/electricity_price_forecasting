
import 'package:flutter/material.dart';

import 'package:front_electricity_forecasting/screens/screens.dart';

class AppRouter {
  static const String initialScreen = "login";

  static Route<dynamic> onGenerateRoute(RouteSettings settings) {
    switch (settings.name) {
      case 'login':
        return MaterialPageRoute(
          builder: (BuildContext context) => const LoginScreen()
        );
      case 'home':
        return RightToLeftPageRoute(
          builder: (BuildContext context) => const HomeScreen(),
          settings: settings
        );
      default:
        return MaterialPageRoute(builder: (_) => const LoginScreen());
    }
  }

}

class RightToLeftPageRoute<T> extends MaterialPageRoute<T> {
  
  RightToLeftPageRoute({builder, settings})
    : super(builder: builder, settings: settings);

  @override
  Widget buildTransitions(BuildContext context, Animation<double> animation, Animation<double> secondaryAnimation, Widget child) {
    return SlideTransition(
      position: Tween<Offset>(
        begin: const Offset(1.0, 0.0), 
        end: const Offset(0.0, 0.0),
      ).animate(animation),
      child: child,
    );
  }
}

class LeftToRightPageRoute<T> extends MaterialPageRoute<T> {
  
  LeftToRightPageRoute({builder, settings})
    : super(builder: builder, settings: settings);

  @override
  Widget buildTransitions(BuildContext context, Animation<double> animation, Animation<double> secondaryAnimation, Widget child) {
    return SlideTransition(
      position: Tween<Offset>(
        begin: const Offset(-1.0, 0.0), 
        end: const Offset(0.0, 0.0),
      ).animate(animation),
      child: child,
    );
  }
}