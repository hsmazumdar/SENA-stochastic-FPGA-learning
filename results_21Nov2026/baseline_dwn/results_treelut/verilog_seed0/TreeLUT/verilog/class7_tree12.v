module class7_tree12(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[49])) | ((~i[349]) & i[49]) | ((~i[251]) & i[335] & i[74] & i[349] & i[49]) | (i[392] & (~i[335]) & i[74] & i[349] & i[49]) | ((~i[77]) & i[320] & (~i[74]) & i[349] & i[49]) | (i[171] & (~i[320]) & (~i[74]) & i[349] & i[49]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
