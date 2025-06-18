
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:front_electricity_forecasting/providers/auth_provider.dart';
import 'package:front_electricity_forecasting/providers/predicts_provider.dart';
import 'package:front_electricity_forecasting/router/app_routes.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const AppState());
}

class AppState extends StatelessWidget {
  const AppState({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (context) => AuthProvider(), lazy: false),
        ChangeNotifierProvider(create: (context) => PredictsProvider(), lazy: false),
      ],
      child: const MyApp(),
    );
  }
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Electricity Price Forecasting',
      initialRoute: AppRouter.initialScreen,
      // routes: AppRouter.routes,
      onGenerateRoute: AppRouter.onGenerateRoute,
    );
  }
}