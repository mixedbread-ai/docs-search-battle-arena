import { z } from "zod";
import { protectedProcedure, router } from "../trpc";

// Input validation schemas
const createDatabaseSchema = z.object({
  label: z.string().min(1),
  provider: z.enum(["algolia", "upstash_search"]),
  credentials: z.string().min(1),
});

export type CreateDatabaseInput = z.infer<typeof createDatabaseSchema>;

const updateDatabaseSchema = z.object({
  id: z.string().uuid(),
  label: z.string().min(1).optional(),
  credentials: z.string().optional(),
});

// Database router
export const databaseRouter = router({
  // Get all databases
  getAll: protectedProcedure.query(async ({ ctx }) => {
    return ctx.databaseService.getAllDatabases();
  }),

  // Get database by ID
  getById: protectedProcedure
    .input(z.object({ id: z.uuid() }))
    .query(async ({ ctx, input }) => {
      return ctx.databaseService.getDatabaseById(input.id);
    }),

  // Create a new database
  create: protectedProcedure
    .input(createDatabaseSchema)
    .mutation(async ({ ctx, input }) => {
      return ctx.databaseService.createDatabase(
        input.label,
        input.provider,
        input.credentials
      );
    }),

  // Update a database
  update: protectedProcedure
    .input(updateDatabaseSchema)
    .mutation(async ({ ctx, input }) => {
      return ctx.databaseService.updateDatabase(input.id, {
        label: input.label,
        credentials: input.credentials,
      });
    }),

  // Delete a database
  delete: protectedProcedure
    .input(z.object({ id: z.string().uuid() }))
    .mutation(async ({ ctx, input }) => {
      return ctx.databaseService.deleteDatabase(input.id);
    }),

  // Test database connection
  testConnection: protectedProcedure
    .input(z.object({ id: z.string().uuid() }))
    .mutation(async ({ ctx, input }) => {
      return ctx.databaseService.testDatabaseConnection(input.id);
    }),
});
