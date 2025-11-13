import Link from "next/link";
import { cn } from "@/src/utils/tailwind";
import { env } from "@/src/env.mjs";

export const LangfuseIcon = ({
  size = 32,
  className,
}: {
  size?: number;
  className?: string;
}) => (
  // eslint-disable-next-line @next/next/no-img-element
  <img
    src={`${env.NEXT_PUBLIC_BASE_PATH ?? ""}/icon.svg`}
    width={size}
    height={size}
    alt="Langfuse Icon"
    className={className}
  />
);

export const LangfuseLogo = ({
  className,
  size = "sm",
  version = false,
}: {
  size?: "sm" | "xl";
  className?: string;
  version?: boolean;
}) => {
  return (
    <div
      className={cn(
        "flex items-center justify-center",
        className,
      )}
    >
      {/* Logo with proper sizing and alignment */}
      <Link href="/" className="flex items-center justify-center">
        <img
          src={`${env.NEXT_PUBLIC_BASE_PATH ?? ""}/icon.svg`}
          alt="Logo"
          className={cn(
            "object-contain block",
            size === "sm" ? "max-h-[28px] w-auto max-w-[110px]" : "h-8 w-auto max-w-[140px]"
          )}
        />
      </Link>
      {/* {version && (
        <VersionLabel className="ml-2 group-data-[collapsible=icon]:hidden" />
      )} */}
    </div>
  );
};
